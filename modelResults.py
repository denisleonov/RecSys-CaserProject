import hashlib
import json


class Results:
    def __init__(self, filename, best_metric='test_NDCG10'):
        self._filename = filename
        self.best_metric = best_metric
        open(self._filename, 'a+')

    def _hash(self, x):
        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save(self, hyperparams, test_metrics, validation_metrics):
        result = {'hash': self._hash(hyperparams)}
        test_dict = {f'test_{k}': v for k, v in test_metrics._asdict().items()}
        validation_dict = {f'val_{k}': v for k, v in validation_metrics._asdict().items()}
        result.update(test_dict)
        result.update(validation_dict)
        result.update(hyperparams)
        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def best(self):
        for x in self:
            print(x)
        results = sorted([x for x in self],
                         key=lambda x: -x[self.best_metric])
        if results:
            return results[0]
        else:
            return None

    def __getitem__(self, hyperparams):
        params_hash = self._hash(hyperparams)
        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)
                if datum['hash'] == params_hash:
                    del datum['hash']
                    return datum
        raise KeyError

    def __contains__(self, x):
        try:
            self[x]
            return True
        except KeyError:
            return False

    def __iter__(self):
        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)
                del datum['hash']
                yield datum
