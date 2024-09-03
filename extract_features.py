import json
import bz2
import sys
import rich
import regex as re
import nltk
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

class PageFeatureExtractor:
    hyphen_word_regex = re.compile(r"(\S*\p{L})-\n(\p{L}\S*)\s?")
    punct_before_regex = re.compile(r"(?<=^|\s)(\p{P}+)(?=\p{L})")
    punct_after_regex = re.compile(r"(?<=\p{L})(\p{P}+)(?=\s|$)")
    zero_width_space_regex = re.compile(r"\u200b")
    max_token_chars = 200
    pos_tag_unknown = "UNK"

    @staticmethod
    def count_longest_alpha_sequence_of_capitalized_lines(lines):
        max_seq_count = 0
        cur_seq_count = 0
        last_char = None

        for line in lines:
            if line and line[0].isupper():
                if last_char is None or line[0] >= last_char:
                    cur_seq_count += 1
                else:
                    max_seq_count = max(cur_seq_count, max_seq_count)
                    cur_seq_count = 1
                last_char = line[0]
            else:
                cur_seq_count = 0

        max_seq_count = max(cur_seq_count, max_seq_count)
        return max_seq_count

    @staticmethod
    def process_text(text):
        text = PageFeatureExtractor.zero_width_space_regex.sub(" ", text)
        text = PageFeatureExtractor.hyphen_word_regex.sub(r"\1\2\n", text)
        text = PageFeatureExtractor.punct_before_regex.sub(r"\1 ", text)
        text = PageFeatureExtractor.punct_after_regex.sub(r" \1", text)
        return text

    @staticmethod
    def extract_basic_section_features(lines):
        non_empty_lines = [PageFeatureExtractor.process_text(line).strip() for line in lines if line.strip()]
        empty_line_count = len(lines) - len(non_empty_lines)

        if not non_empty_lines:
            return None

        begin_char_count = Counter(line[0] for line in non_empty_lines)
        end_char_count = Counter(line[-1] for line in non_empty_lines)
        longest_alpha_seq = PageFeatureExtractor.count_longest_alpha_sequence_of_capitalized_lines(non_empty_lines)

        text = "\n".join(non_empty_lines)
        if not text:
            return None

        tokens = word_tokenize(text)
        token_pos = [(token[:PageFeatureExtractor.max_token_chars], PageFeatureExtractor.pos_tag_unknown) for token in tokens]
        token_pos_count = defaultdict(lambda: defaultdict(int))
        for token, pos in token_pos:
            token_pos_count[token][pos] += 1

        return {
            "tokenCount": len(token_pos),
            "lineCount": len(lines),
            "emptyLineCount": empty_line_count,
            "capAlphaSeq": longest_alpha_seq,
            "beginCharCount": dict(begin_char_count),
            "endCharCount": dict(end_char_count),
            "tokenPosCount": {token: dict(pos_counts) for token, pos_counts in token_pos_count.items()}
        }


#with bz2.open('data.json.bz2', 'rt') as f:
#    data = json.load(f)
#    page_data = data['features']['pages'][0]['body']
#    rich.print(page_data)
input_file = 'fodor.txt'
output_file = 'fodorrEXTRACTED.json.bz2'


with open(input_file, 'r') as f:
    txt = f.read().strip()
    lines = txt.split('\n')
    #rich.print(lines)
    features = PageFeatureExtractor.extract_basic_section_features(lines)
    data = {'features': {'pages': [{'body': features, 'original': txt}]}}
    #rich.print(features)
    with bz2.open(output_file, 'wt') as f:
        json.dump(data, f, ensure_ascii=False)