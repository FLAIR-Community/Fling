from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset

from fling.utils.registry_utils import DATASET_REGISTRY


@DATASET_REGISTRY('ag_news')
class AGNewsDataset(Dataset):
    def __init__(self, cfg: dict, train: bool):
        super(AGNewsDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        split = 'train' if self.train else 'test'
        self.dataset = AG_NEWS(cfg.data.data_path, split=split)
        self.tokenizer = get_tokenizer("basic_english")
        self.max_length = cfg.data.max_length

        # Prepare vocabulary tabular.
        def _yield_tokens(data_iter):
            for _, text in data_iter:
                yield self.tokenizer(text)

        self.vocab = build_vocab_from_iterator(_yield_tokens(iter(self.dataset)), speicals=['<unk>', '<pad>'])
        self.vocab.set_default_index(self.vocab["<unk>"])

        self.process_text = lambda x: self.vocab(self.tokenizer(x))
        self.process_label = lambda x: int(x) - 1
        print(f'Dataset Generated. Total vocabs: {len(self.vocab)}')

    def __getitem__(self, item):
        label, text = self.dataset[item]
        label = self.process_label(label)
        text = self.process_text(text)

        if len(text) > self.max_length:
            text = text[:self.max_length]
        else:
            text += [self.vocab['<pad>']] * (self.max_length - len(text))

        return {'input': text, 'class_id': label}
