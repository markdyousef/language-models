from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from glob import glob

def get_babi_task(dpath, task_id):
    fpaths = glob(f'{dpath}/{qa}_*')
    for fpath in fpaths:
        if 'train' in fpath:
            with open(fpath, 'r') as fp:
                train = fp.read()
        elif 'test' in fpath:
            with open (fpath, 'r') as fp:
                test = fp.read()
    return train, test

def get_unindexed_qa(raw):
    tasks, task = [], None
    lines = raw.strip().split('\n')
    for i, line in enumerate(lines):
        idx = int(line[0:line.find(' ')])
        if idx == 1:
            # context, question, answer, supporting facts
            task = {'C': '','Q': '', 'A': '', 'S': ''}
            count = 0
            id_map = {}
        
        line = line.strip().replace('.', ' . ')[line.find(' ')+1:]
        # not a question
        if line.find('?') == -1:
            task['C'] += line+'<line>'
            id_map[idx] = count
            count += 1
        else:
            qidx = line.find('?')
            tmp = line[qidx+1:].sxplit('\t')
            task['Q'] = line[:qidx]
            task['A'] = tmp[1].strip()
            task['S'] = [id_map[int(o.strip())] for o in tmp[2]]
            tc = task.copy()
            tc['C'] = tc['C'].split('<line>')[:-1]
            tasks.append(tc)
    return tasks

def format_sentence(sent):
    return sent.lower().split()+['<EOS>']



# adapted from https://github.com/dandelin/Dynamic-memory-networks-plus-Pytorch
class BabiDataset(Dataset):
    def __init__(self, dpath, task_id, mode='train'):
        self.vocab_path = f'dpath/babi{tast_id}_vocab.pkl'
        self.mode = mode
        self.vocab = {'<PAD>': 0, '<EOS>': 1}
        train_raw, test_raw = get_babi_task(task_id)
        self.train = self.index_task(train_raw)
        self.test = self.index_task(test_raw)
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train[0])
        elif self.mode == 'test':
            return len(self.test[0])
    
    def __getitem__(self, index):
        if self.mode == 'train':
            contexts, questions, answers = self.train
        elif self.mode == 'test':
            contexts, questions, answers = self.test
        return contexts[index], questions[index], answers[index]

    def build_vocab(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
    
    def index_task(self, raw):
        unindexed = get_unindexed_qa(raw)
        contexts, questions, answers = [],  [], []
        for qa in unindexed:
            # context
            context = [format_sentence(c) for c in qa['C']]
            for c in context:
                for token in c: self.build_vocab(token)
            context = [[self.vocab[token] for token in sent] for sent in context]
            #question
            question = format_sentence(qa['Q'])
            for token in question: self.build_vocab(token)
            question = [self.vocab[token] for token in question]
            # anwer
            self.build_vocab(qa['A'.lower()])
            anwer = self.vocab[qa['A'].lower()]

            contexts.append(context)
            questions.append(question)
            answers.append(anwer)
        return (contexts, questions, answers)

