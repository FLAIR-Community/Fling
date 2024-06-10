import torch
from torch.utils.data import Dataset
from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('ag_news')
class AGNewsDataset(Dataset):
    """
    Implementation of AG news dataset. This dataset contains over 1 million of news articles with 4 categories.
    For more information, please refer to the link: http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .
    """
    vocab = None

    def __init__(self, cfg: dict, train: bool):
        try:
            from torchtext.datasets import AG_NEWS
            from torchtext.data.utils import get_tokenizer
            from torchtext.vocab import build_vocab_from_iterator
        except ImportError:
            raise ImportError('torchtext is not successfully installed. Please install it manually.')

        super(AGNewsDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        split = 'train' if self.train else 'test'
        self.dataset = list(AG_NEWS(cfg.data.data_path, split=split))
        self.tokenizer = get_tokenizer("basic_english")
        self.max_length = cfg.data.get('max_length', float('inf'))

        def _yield_tokens(data_iter):
            for _, text in data_iter:
                dat = self.tokenizer(text)
                yield dat

        # Prepare vocabulary tabular.
        if AGNewsDataset.vocab is None:
            AGNewsDataset.vocab = build_vocab_from_iterator(
                _yield_tokens(iter(self.dataset)), specials=['<unk>', '<pad>'], min_freq=5
            )
            AGNewsDataset.vocab.set_default_index(self.vocab["<unk>"])

        real_max_len = max([len(self._process_text((self.dataset[i][1]))) for i in range(len(self.dataset))])
        self.max_length = min(self.max_length, real_max_len)

        print(
            f'Dataset Generated. Total vocab size: {len(self.vocab)}; '
            f'Max length of the input: {self.max_length}; '
            f'Dataset length: {len(self.dataset)}.'
        )

    def _process_text(self, x):
        return AGNewsDataset.vocab(self.tokenizer(x))

    def _process_label(self, x):
        return int(x) - 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        label, text = self.dataset[item]
        label = self._process_label(label)
        text = self._process_text(text)

        if len(text) > self.max_length:
            text = text[:self.max_length]
        else:
            text += [self.vocab['<pad>']] * (self.max_length - len(text))

        assert len(text) == self.max_length
        return {'input': torch.LongTensor(text), 'class_id': label}
