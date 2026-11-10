"""Every dataset this project reads. No dataset ID is hard-coded anywhere else.

Rule (owner, 2026-11-02): only established releases from known orgs with papers
(AI4Bharat, Google official, Meta, Cohere). No one-person HF copies.

role: tamil / english / tanglish_real feed pretraining; sft feeds the SFT stage;
eval and heldout are never trained on (their sentences go into the blocklist).
"""
from dataclasses import dataclass
from typing import Literal

Role = Literal["tamil", "english", "tanglish_real", "code", "sft", "eval", "heldout"]


@dataclass(frozen=True)
class Source:
    hf_id: str
    config: str | None
    split: str
    text_field: str  # comma-separated when several fields hold text
    role: Role
    license: str
    data_dir: str | None = None
    data_files: str | None = None
    url: str | None = None  # non-HF official release: a .tar fetched once
    member: str | None = None  # file inside that tar/zip
    revision: str | None = None  # e.g. refs/convert/parquet: HF's own parquet copy of a script-based dataset
    skip_docs: int = 0  # drop the first N non-empty docs (the held-out ones), same counting as load_heldout

    @property
    def fields(self) -> list[str]:
        return [f.strip() for f in self.text_field.split(",")]


# Google's official Dakshina release (Roark et al., LREC 2020)
DAKSHINA_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
_DAK = "dakshina_dataset_v1.0/ta/romanized/ta.romanized.rejoined.{}.roman.txt"

# pretraining
SANGRAHA_TA = Source("ai4bharat/sangraha", None, "train", "text", "tamil", "cc-by-4.0", data_dir="verified/tam")
INDICCORP_TA = Source("ai4bharat/IndicCorpV2", None, "train", "text", "tamil", "cc0-1.0", data_files="data/ta.txt")
FINEWEB_EDU_EN = Source("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", "english", "odc-by")
DAKSHINA_TA = Source("google/dakshina", None, "dev", "text", "tanglish_real", "cc-by-sa-4.0",
                     url=DAKSHINA_URL, member=_DAK.format("dev"))

# code: Hugging Face's codeparrot GitHub Code (clean), apache-2.0, ungated; 32 languages mixed in every file.
# Tokenizer v2 only (owner decision 2026-11-03: light code support, ~5% of the tokenizer mix); 4 of 880 files is plenty.
# read with the generic parquet builder: the repo also ships a loading script, which `datasets` >= 4 refuses
GITHUB_CODE = Source("parquet", None, "train", "code", "code", "apache-2.0",
                     data_files="hf://datasets/codeparrot/github-code-clean/data/train-0000[0-3]-of-00880.parquet")
CODE = [GITHUB_CODE]

# fertility/eval read the first HELDOUT_DOCS docs of each held-out set; exactly those are blocklisted
HELDOUT_DOCS = 2000
# Tamil Wikipedia minus its first HELDOUT_DOCS articles (the exam); owner decision 2026-11-03
WIKI_TA_TRAIN = Source("wikimedia/wikipedia", "20231101.ta", "train", "text", "tamil", "cc-by-sa-3.0",
                       skip_docs=HELDOUT_DOCS)

# SFT
AYA_TA = Source("CohereLabs/aya_dataset", None, "train", "inputs,targets", "sft", "apache-2.0")
# IndicAlign subsets have different schemas, so each is loaded from its own parquet file(s) (a whole-folder load
# mixes them and the first file's columns win). Dolly_T / OpenAssistant_T / Wiki_Chat keep Tamil in column
# `tam_Taml` as [[user, assistant], ...]; Anudesh has one mixed-language `interactions` column (~2k Tamil of 36.8k).
_IA = "indicalign-instruct/"
INDICALIGN_ANUDESH_TA = Source("ai4bharat/indic-align", None, "train", "interactions", "sft", "cc-by-4.0",
                               data_files=_IA + "anudesh/anudesh1.parquet")
INDICALIGN_DOLLY_TA = Source("ai4bharat/indic-align", None, "train", "tam_Taml", "sft", "cc-by-4.0",
                             data_files=_IA + "dolly/Dolly.parquet")  # ~14.9k Tamil
INDICALIGN_OASST_TA = Source("ai4bharat/indic-align", None, "train", "tam_Taml", "sft", "cc-by-4.0",
                             data_files=_IA + "oasst/oasst.parquet")  # ~19.8k Tamil
INDICALIGN_WIKICHAT_TA = Source("ai4bharat/indic-align", None, "train", "tam_Taml", "sft", "cc-by-4.0",
                                data_files=_IA + "wiki_chat/wiki_chat_000[0-2]_of_0031.parquet")  # 3 of 32 shards, ~18.5k
INDICALIGN = [INDICALIGN_ANUDESH_TA, INDICALIGN_DOLLY_TA, INDICALIGN_OASST_TA, INDICALIGN_WIKICHAT_TA]

# held out (fertility benchmark, bits-per-byte)
WIKI_TA = Source("wikimedia/wikipedia", "20231101.ta", "train", "text", "heldout", "cc-by-sa-3.0")
WIKI_EN = Source("wikimedia/wikipedia", "20231101.en", "train", "text", "heldout", "cc-by-sa-3.0")
DAKSHINA_TA_TEST = Source("google/dakshina", None, "test", "text", "heldout", "cc-by-sa-4.0",
                          url=DAKSHINA_URL, member=_DAK.format("test"))

# labeled eval
# These three ship a loading script, which `datasets` >= 4 refuses to run. HF's own parquet
# conversion of the same data (refs/convert/parquet) loads without it.
_PQ = "refs/convert/parquet"
# official release by the IndicXNLI paper's first author (arXiv 2204.08776); ai4bharat/IndicXNLI does not exist
INDICXNLI_TA = Source("Divyanshu/indicxnli", None, "test", "premise,hypothesis", "eval", "cc0-1.0",
                      data_files="ta/test/0000.parquet", revision=_PQ)
INDICSENTIMENT_TA = Source("ai4bharat/IndicSentiment", None, "test", "INDIC REVIEW", "eval", "cc0-1.0",
                           data_files="translation-ta/test/0000.parquet", revision=_PQ)
INDICQA_TA = Source("ai4bharat/IndicQA", None, "test", "context,question", "eval", "cc-by-4.0",
                    data_files="indicqa.ta/test/0000.parquet", revision=_PQ)
BELEBELE_TA = Source("facebook/belebele", "tam_Taml", "test",
                     "flores_passage,question,mc_answer1,mc_answer2,mc_answer3,mc_answer4", "eval", "cc-by-sa-4.0")
# Aksharantar's repo is one zip per language with JSON-lines files inside (no HF configs)
AKSHARANTAR_TA = Source("ai4bharat/Aksharantar", None, "test", "native word,english word", "eval", "cc-by-4.0",
                        url="https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/tam.zip",
                        member="tam_test.json")

PRETRAIN = [SANGRAHA_TA, INDICCORP_TA, WIKI_TA_TRAIN, FINEWEB_EDU_EN, DAKSHINA_TA]
SFT = [AYA_TA, *INDICALIGN]
HELDOUT = [WIKI_TA, WIKI_EN, DAKSHINA_TA_TEST]
EVAL = [INDICXNLI_TA, INDICSENTIMENT_TA, INDICQA_TA, BELEBELE_TA, AKSHARANTAR_TA]
