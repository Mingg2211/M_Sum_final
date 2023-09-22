# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import re
import string
import zipfile

import nltk
from underthesea import sent_tokenize
from .._compat import to_string, to_unicode, unicode
from ..utils import normalize_language

class DefaultWordTokenizer(object):
    """NLTK tokenizer"""
    def tokenize(self, text):
        return nltk.word_tokenize(text)


class ChineseWordTokenizer:
    def tokenize(self, text):
        try:
            import jieba
        except ImportError as e:
            raise ValueError("Chinese tokenizer requires jieba. Please, install it by command 'pip install jieba'.")
        return jieba.cut(text)

class VietNamWordTokenizer:
    def tokenize(self, text):
        try:
            from underthesea import word_tokenize
        except ImportError as e:
            raise ValueError("VietNam tokenizer requires underthesea. Please, install it by command 'pip install underthesea'.")
        return word_tokenize(text)
    
class RussianWordTokenizer:
    def tokenize(self, text):
        try:
            import nltk
        except ImportError as e:
            raise ValueError("Russian tokenizer requires nltk. Please, install it by command 'pip install nltk'.")
        return nltk.word_tokenize(text, language='russian')
class VietNameseSentencesTokenizer:
    def tokenize(self, text):
        if '\n\n'  not in text:
            return sent_tokenize(text)
        else:
            list_sent = text.split('\n\n')
            return list_sent

class ChineseSentencesTokenizer():
    def tokenize(self, text):
        if '\n\n'  not in text:
            list_sent = [s.replace('\u3000',' ').strip() for s in re.split('[。！？；]', text)]
            return list_sent
        else:
            list_sent = text.split('\n\n')
            return list_sent

class Tokenizer(object):
    """Language dependent tokenizer of text document."""

    _WORD_PATTERN = re.compile(r"^[^\W\d_](?:[^\W\d_]|['-])*$", re.UNICODE)
    # feel free to contribute if you have better tokenizer for any of these languages :)
    LANGUAGE_ALIASES = {
        "slovak": "czech",
    }

    # improve tokenizer by adding specific abbreviations it has issues with
    # note the final point in these items must not be included
    LANGUAGE_EXTRA_ABREVS = {
        "english": ["e.g", "al", "i.e"],
        "russian": ["ім.", "о.", "вул.", "просп.", "бул.", "пров.", "пл.", "г.", "р.", "див.", "п.", "с.", "м."],
        "vietnamese": ['v.v'],
    }

    SPECIAL_SENTENCE_TOKENIZERS = {
        'russian': VietNameseSentencesTokenizer(),
        'english': VietNameseSentencesTokenizer(),
        'vietnamese' : VietNameseSentencesTokenizer(),
        'chinese': ChineseSentencesTokenizer(),
    }

    SPECIAL_WORD_TOKENIZERS = {

        'chinese': ChineseWordTokenizer(),
        'vietnamese': VietNamWordTokenizer(),
        'russian':RussianWordTokenizer(),
    }

    def __init__(self, language):
        language = normalize_language(language)
        self._language = language

        tokenizer_language = self.LANGUAGE_ALIASES.get(language, language)
        self._sentence_tokenizer = self._get_sentence_tokenizer(tokenizer_language)
        self._word_tokenizer = self._get_word_tokenizer(tokenizer_language)

    @property
    def language(self):
        return self._language

    def _get_sentence_tokenizer(self, language):
        if language in self.SPECIAL_SENTENCE_TOKENIZERS:
            return self.SPECIAL_SENTENCE_TOKENIZERS[language]

    def _get_word_tokenizer(self, language):
        if language in self.SPECIAL_WORD_TOKENIZERS:
            return self.SPECIAL_WORD_TOKENIZERS[language]
        else:
            return DefaultWordTokenizer()

    
    def flatten_list(self, list):
        tmp = (" ".join([str(item) for item in list]))
        res = []
        return res.append(tmp)

    def to_sentences(self, paragraph):
        if hasattr(self._sentence_tokenizer, '_params'):
            extra_abbreviations = self.LANGUAGE_EXTRA_ABREVS.get(self._language, [])
            self._sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
        sentences = self._sentence_tokenizer.tokenize(to_unicode(paragraph))
        # print('----------')
        # print(sentences)
        # print(type(sentences))
        return tuple(map(unicode.strip, sentences))

    def to_words(self, sentence):
        words = self._word_tokenizer.tokenize(to_unicode(sentence))
        return tuple(filter(self._is_word, words))

    @staticmethod
    def _is_word(word):
        return bool(Tokenizer._WORD_PATTERN.match(word))


# mingg = Tokenizer(language='vietnamese')
# doc = """
# PGS.TS Lê Hoàng Sơn công bố hơn 180 công trình, bài báo trên các tạp chí nước ngoài trong danh mục ISI. Ông là gương mặt lọt vào top 10.000 nhà khoa học xuất sắc của thế giới trong 4 năm liên tiếp 2019, 2020, 2021, 2022, đồng thời được gắn huy hiệu "Rising Star" - ngôi sao khoa học đang lên xuất sắc trên thế giới năm 2022.\n\nLĩnh vực Kỹ thuật và Công nghệ tiếp tục có GS.TSKH Nguyễn Đình Đức, ĐHQGHN. Ông là một trong những nhà khoa học đầu ngành của Việt Nam trong lĩnh vực Cơ học và vật liệu composite. Ông đã công bố trên 300 công trình khoa học, trong đó có 200 bài trên các tạp chí quốc tế ISI có uy tín. Bốn năm liên tiếp 2019, 2020, 2021 và 2022 ông lọt vào top 100.000 nhà khoa học có ảnh hưởng nhất thế giới. GS. Nguyễn Đình Đức vào tốp 94 thế giới trong lĩnh vực Engineering năm 2022, tức tốp 100 thế giới.\n\nLĩnh vực Khoa học Môi trường có GS.TS Phạm Hùng Việt và PGS.TS Từ Bình Minh, đều từ Trường Đại học Khoa học Tự nhiên, ĐHQGHN. GS. Phạm Hùng Việt hiện là Giám đốc Phòng thí nghiệm trọng điểm Công nghệ phân tích phục vụ kiểm định môi trường và An toàn thực phẩm, Trưởng nhóm nghiên cứu mạnh. Ông có hơn 100 công trình, bài báo công bố, sở hữu nhiều bằng sáng chế.\n\nPGS.TS Từ Bình Minh là nhà khoa học trong lĩnh vực hóa học. Chỉ trong hai năm 2019, 2020, nhóm nghiên cứu của ông đã công bố trên 20 công trình đăng trên các tạp chí quốc tế thuộc danh mục ISI uy tín, nhiều tạp chí trong số đó thuộc TOP 5% theo lĩnh vực chuyên sâu. Năm 2022, PGS.TS Từ Bình Minh cũng vào top nhà khoa học có ảnh hưởng nhất thế giới.
# """ 

# print(len(mingg.to_sentences(paragraph=doc)))