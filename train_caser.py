import argparse
from tqdm.auto import tqdm

import torch.optim as optim
from attrdict import AttrDict
import wandb
from torch.nn.parallel import data_parallel

from caser import Caser
from evaluation import evaluate_ranking
from interactions import Interactions
from utils import *


class Recommender(object):
    """
    Contains attributes and methods that needed to train a sequential
    recommendation model. Models are trained by many tuples of
    (users, sequences, targets, negatives) and negatives are from negative
    sampling: for any known tuple of (user, sequence, targets), one or more
    items are randomly sampled to act as negatives.


    Parameters
    ----------

    n_iter: int,
        Number of iterations to run.
    batch_size: int,
        Minibatch size.
    l2: float,
        L2 loss penalty, also known as the 'lambda' of l2 regularization.
    neg_samples: int,
        Number of negative samples to generate for each targets.
        If targets=3 and neg_samples=3, then it will sample 9 negatives.
    learning_rate: float,
        Initial learning rate.
    use_cuda: boolean,
        Run the model on a GPU or CPU.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 learning_rate=None,
                 use_cuda=False,
                 model_args=None,
                 ngpus=None,
                 ):

        # model related
        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args

        # learning related
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._ngpus = torch.cuda.device_count() if ngpus is None else ngpus

        # rank evaluation related
        self.test_sequence = None
        self._candidate = dict()

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        self._net = Caser(self._num_users,
                          self._num_items,
                          self.model_args).to(self._device)

        self._optimizer = optim.Adam(self._net.parameters(),
                                     weight_decay=self._l2,
                                     lr=self._learning_rate)

    def fit(self, train, test, verbose=False):
        """
        The general training loop to fit the model

        Parameters
        ----------

        train: :class:`spotlight.interactions.Interactions`
            training instances, also contains test sequences
        test: :class:`spotlight.interactions.Interactions`
            only contains targets for test sequences
        verbose: bool, optional
            print the logs
        """

        # convert to sequences, targets and users
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids.reshape(-1, 1)

        n_train = sequences_np.shape[0]

        print('total training instances: %d' % n_train)

        if not self._initialized:
            self._initialize(train)

        epoch_pbar = tqdm(range(0, self._n_iter), total=self._n_iter)
        for epoch_num in epoch_pbar:

            # set model to training mode
            self._net.train()

            users_np, sequences_np, targets_np = shuffle(users_np,
                                                         sequences_np,
                                                         targets_np)

            negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)

            # convert numpy arrays to PyTorch tensors and move it to the corresponding devices
            users, sequences, targets, negatives = (torch.from_numpy(users_np).long(),
                                                    torch.from_numpy(sequences_np).long(),
                                                    torch.from_numpy(targets_np).long(),
                                                    torch.from_numpy(negatives_np).long())

            users, sequences, targets, negatives = (users.to(self._device),
                                                    sequences.to(self._device),
                                                    targets.to(self._device),
                                                    negatives.to(self._device))

            epoch_loss = 0.0
            batch_pbar = tqdm(enumerate(minibatch(users, sequences, targets, negatives, batch_size=self._batch_size)),
                              total=len(users) // self._batch_size, leave=True)

            for (minibatch_num, (batch_users, batch_sequences, batch_targets, batch_negatives)) in batch_pbar:
                items_to_predict = torch.cat((batch_targets, batch_negatives), 1)

                if self._ngpus > 1:
                    items_prediction = data_parallel(self._net, (batch_sequences, batch_users, items_to_predict),
                                                     device_ids=list(range(self._ngpus)))
                else:
                    items_prediction = self._net(batch_sequences,
                                                 batch_users,
                                                 items_to_predict)

                (targets_prediction,
                 negatives_prediction) = torch.split(items_prediction,
                                                     [batch_targets.size(1),
                                                      batch_negatives.size(1)], dim=1)

                self._optimizer.zero_grad()
                # compute the binary cross-entropy loss
                positive_loss = -torch.mean(
                    torch.log(torch.sigmoid(targets_prediction)))
                negative_loss = -torch.mean(
                    torch.log(1 - torch.sigmoid(negatives_prediction)))
                loss = positive_loss + negative_loss

                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1

            if verbose and (epoch_num + 1) % 2 == 0:
                precision, recall, ndcgs, mean_aps = evaluate_ranking(self, test, train, k=[1, 5, 10])
                logdict = {
                    'loss': epoch_loss,
                    'map': mean_aps,
                    'prec@1': np.mean(precision[0]),
                    'prec@5': np.mean(precision[1]),
                    'prec@10': np.mean(precision[2]),
                    'recall@1': np.mean(recall[0]),
                    'recall@5': np.mean(recall[1]),
                    'recall@10': np.mean(recall[2]),
                    'ndcg@1': np.mean(ndcgs[0]),
                    'ndcg@5': np.mean(ndcgs[1]),
                    'ndcg@10': np.mean(ndcgs[2]),
                }
                wandb.log(logdict, step=epoch_num)


    def _generate_negative_samples(self, users, interactions, n):
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} / {Items Rated by User}

        Parameters
        ----------

        users: array of np.int64
            sequence users
        interactions: :class:`spotlight.interactions.Interactions`
            training instances, used for generate candidates
        n: int
            total number of negatives to sample for each sequence
        """

        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items - 1) + 1  # 0 for padding
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[
                    np.random.randint(len(x))]

        return negative_samples

    def predict(self, user_id, item_ids=None):
        """
        Make predictions for evaluation: given a user id, it will
        first retrieve the test sequence associated with that user
        and compute the recommendation scores for items.

        Parameters
        ----------

        user_id: int
           users id for which prediction scores needed.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        """

        if self.test_sequence is None:
            raise ValueError('Missing test sequences, cannot make predictions')

        # set model to evaluation model
        self._net.eval()
        with torch.no_grad():
            sequences_np = self.test_sequence.sequences[user_id, :]
            sequences_np = np.atleast_2d(sequences_np)

            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1, 1)

            sequences = torch.from_numpy(sequences_np).long()
            item_ids = torch.from_numpy(item_ids).long()
            user_id = torch.from_numpy(np.array([[user_id]])).long()

            user, sequences, items = (user_id.to(self._device),
                                      sequences.to(self._device),
                                      item_ids.to(self._device))

            out = self._net(sequences,
                            user,
                            items,
                            for_pred=True)

        return out.cpu().numpy().flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--train_root', type=str, default='datasets/ml1m/train.txt')
    parser.add_argument('--test_root', type=str, default='datasets/ml1m/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--n_gpus', type=int, default=None)
    parser.add_argument('--wandb', type=str, default='test')

    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--nv', type=int, default=4)
    parser.add_argument('--nh', type=int, default=16)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--ac_conv', type=str, default='relu')
    parser.add_argument('--ac_fc', type=str, default='relu')

    config = parser.parse_args()

    model_config = {
        'L': config.L,
        'd': config.d,
        'nv': config.nv,
        'nh': config.nh,
        'drop': config.drop,
        'ac_conv': config.ac_conv,
        'ac_fc': config.ac_fc,
    }
    model_config = AttrDict(model_config)

    # set seed
    set_seed(config.seed,
             cuda=config.use_cuda)

    # wandb
    wandb.init(
        config=config,
        project="Caser",
        name=config.wandb,
    )

    # load dataset
    train = Interactions(config.train_root)
    # transform triplets to sequence representation
    train.to_sequence(config.L, config.T)

    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    print(config)
    # fit model
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        model_args=model_config,
                        use_cuda=config.use_cuda,
                        ngpus=config.n_gpus)

    model.fit(train, test, verbose=True)
