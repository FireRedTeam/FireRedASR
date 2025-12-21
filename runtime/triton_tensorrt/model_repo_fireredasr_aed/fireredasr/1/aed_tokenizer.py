import logging
import re

import sentencepiece as spm


class TokenDict:
    def __init__(self, dict_path, unk=""):
        assert dict_path != ""
        self.id2word, self.word2id = self.read_dict(dict_path)
        self.unk = unk
        assert unk == "" or unk in self.word2id
        self.unkid = self.word2id[unk] if unk else -1

    def get(self, key, default):
        if type(default) == str:
            default = self.word2id[default]
        return self.word2id.get(key, default)

    def __getitem__(self, key):
        if type(key) == str:
            if self.unk:
                return self.word2id.get(key, self.word2id[self.unk])
            else:
                return self.word2id[key]
        elif type(key) == int:
            return self.id2word[key]
        else:
            raise TypeError("Key should be str or int")

    def __len__(self):
        return len(self.id2word)

    def __contains__(self, query):
        if type(query) == str:
            return query in self.word2id
        elif type(query) == int:
            return query in self.id2word
        else:
            raise TypeError("query should be str or int")

    def read_dict(self, dict_path):
        id2word, word2id = [], {}
        with open(dict_path, encoding='utf8') as f:
            for i, line in enumerate(f):
                tokens = line.strip().split()
                if len(tokens) >= 2:
                    word, index = tokens[0], int(tokens[1])
                elif len(tokens) == 1:
                    word, index = tokens[0], i
                else:  # empty line or space
                    logging.info(f"Find empty line or space '{line.strip()}' in {dict_path}:L{i}, set to ' '")
                    word, index = " ", i
                assert len(id2word) == index
                assert len(word2id) == index
                if word == "<space>":
                    logging.info(f"NOTE: Find <space> in {dict_path}:L{i} and convert it to ' '")
                    word = " "
                word2id[word] = index
                id2word.append(word)
        assert len(id2word) == len(word2id)
        return id2word, word2id



class ChineseCharEnglishSpmTokenizer:
    """
    - One Chinese char is a token.
    - Split English word into SPM and one piece is a token.
    - Ignore ' ' between Chinese char
    - Replace ' ' between English word with "▁" by spm_model
    - Need to put SPM piece into dict file
    - If not set spm_model, will use English char and <space>
    """
    SPM_SPACE = "▁"

    def __init__(self, dict_path, spm_model, unk="<unk>", space="<space>"):
        self.dict = TokenDict(dict_path, unk=unk)
        self.space = space
        if spm_model:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(spm_model)
        else:
            self.sp = None
            print("[WRAN] Not set spm_model, will use English char")
            print("[WARN] Please check how to deal with ' '(space)")
            if self.space not in self.dict:
                print("Please add <space> to your dict, or it will be <unk>")

    def tokenize(self, text, replace_punc=True):
        text = text.upper()
        tokens = []
        if replace_punc:
            text = re.sub("[，。？！,\.?!]", " ", text)
        pattern = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff])')
        parts = pattern.split(text.strip())
        parts = [p for p in parts if len(p.strip()) > 0]
        for part in parts:
            if pattern.fullmatch(part) is not None:
                tokens.append(part)
            else:
                if self.sp:
                    for piece in self.sp.EncodeAsPieces(part.strip()):
                        tokens.append(piece)
                else:
                    for char in part.strip():
                        tokens.append(char if char != " " else self.space)
        tokens_id = []
        for token in tokens:
            tokens_id.append(self.dict.get(token, self.dict.unk))
        return tokens, tokens_id

    def detokenize(self, inputs, join_symbol="", replace_spm_space=True):
        """inputs is ids or tokens, do not need self.sp"""
        if len(inputs) > 0 and type(inputs[0]) == int:
            tokens = [self.dict[id] for id in inputs]
        else:
            tokens = inputs
        s = f"{join_symbol}".join(tokens)
        if replace_spm_space:
            s = s.replace(self.SPM_SPACE, ' ').strip()
        return s
