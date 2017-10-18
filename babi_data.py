from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from glob import glob

def pad_collate(batch):
    max_context_sen_len = float('-inf')
    max_context_len = float('-inf')
    max_question_len = float('-inf')
    for elem in batch:
        context, question, _ = elem
        max_context_len = max_context_len if max_context_len > len(context) else len(context)
        max_question_len = max_question_len if max_question_len > len(question) else len(question)
        for sen in context:
            max_context_sen_len = max_context_sen_len if max_context_sen_len > len(sen) else len(sen)
    max_context_len = min(max_context_len, 70)
    for i, elem in enumerate(batch):
        _context, question, answer = elem
        _context = _context[-max_context_len:]
        context = np.zeros((max_context_len, max_context_sen_len))
        for j, sen in enumerate(_context):
            context[j] = np.pad(sen, (0, max_context_sen_len - len(sen)), 'constant', constant_values=0)
        question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
        batch[i] = (context, question, answer)
    return default_collate(batch)

def get_babi_task(dpath, task_id):
    fpaths = glob(f'{dpath}/qa{task_id}_*')
    if not fpaths: print('No files')
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
            tmp = line[qidx+1:].split('\t')
            task['Q'] = line[:qidx]
            task['A'] = tmp[1].strip()
            task['S'] = [id_map[int(o.strip())] for o in tmp[2].split()]
            tc = task.copy()
            tc['C'] = tc['C'].split('<line>')[:-1]
            tasks.append(tc)
    return tasks

def format_sentence(sent):
    return sent.lower().split()+['<EOS>']



# adapted from https://github.com/dandelin/Dynamic-memory-networks-plus-Pytorch
class BabiDataset(Dataset):
    def __init__(self, dpath='/home/mark/data/datasets/nlp/babi/tasks_1-20_v1-2/en-10k', task_id=1, mode='train'):
        self.vocab_path = f'dpath/babi{task_id}_vocab.pkl'
        self.mode = mode
        self.vocab = {'<PAD>': 0, '<EOS>': 1}
        train_raw, test_raw = get_babi_task(dpath, task_id)
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

    def set_mode(self, mode):
        self.mode = mode

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
            self.build_vocab(qa['A'].lower())
            anwer = self.vocab[qa['A'].lower()]

            contexts.append(context)
            questions.append(question)
            answers.append(anwer)
        return (contexts, questions, answers)

