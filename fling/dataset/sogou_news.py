import csv

import torch
from torch.utils.data import Dataset

from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY.register('sogou_news')
class SogouNewsDataset(Dataset):
    """
    Implementation of Sogou news dataset. The Sogou News dataset is a mixture of 2,909,551 news articles from the \
    SogouCA and SogouCS news corpora, in 5 categories. For more information, please refer to the link: \
    http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .
    """
    vocab = None

    def __init__(self, cfg: dict, train: bool):
        try:
            from torchtext.datasets import SogouNews
            from torchtext.data.utils import get_tokenizer
            from torchtext.vocab import build_vocab_from_iterator
        except ImportError:
            raise ImportError('torchtext is not successfully installed. Please install it manually.')

        super(SogouNewsDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        split = 'train' if self.train else 'test'
        csv.field_size_limit(int(1e8))
        self.dataset = list(SogouNews(cfg.data.data_path, split=split))
        self.tokenizer = get_tokenizer("basic_english")
        self.max_length = cfg.data.get('max_length', float('inf'))

        def _yield_tokens(data_iter):
            for _, text in data_iter:
                dat = self.tokenizer(text)
                yield dat

        # Prepare vocabulary tabular.
        if SogouNewsDataset.vocab is None:
            SogouNewsDataset.vocab = build_vocab_from_iterator(
                _yield_tokens(iter(self.dataset)), specials=['<unk>', '<pad>'], min_freq=5
            )
            SogouNewsDataset.vocab.set_default_index(self.vocab["<unk>"])

        real_max_len = max([len(self._process_text((self.dataset[i][1]))) for i in range(len(self.dataset))])
        self.max_length = min(self.max_length, real_max_len)

        print(
            f'Dataset Generated. Total vocab size: {len(self.vocab)}; '
            f'Max length of the input: {self.max_length}; '
            f'Dataset length: {len(self.dataset)}.'
        )

    def _process_text(self, x):
        return SogouNewsDataset.vocab(self.tokenizer(x))

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
