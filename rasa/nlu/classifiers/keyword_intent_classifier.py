import os
import logging
import re
from typing import Any, Dict, Optional, Text

from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.nlu import utils
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.constants import (
    INTENT,
    METADATA,
    METADATA_EXAMPLE,
    METADATA_INTENT,
    TEXT,
)

import rasa.shared.utils.io
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.model import Metadata

logger = logging.getLogger(__name__)


class KeywordIntentClassifier(IntentClassifier):
    """Intent classifier looking for keyword matches in the training examples
    with the ability to customise the match condition, and override an existing
    classification if a keyword intent is misclassified by any prior NLU classifier.

    The classifier takes a list of keywords, match conditions and associated intents
    as an input. An input sentence is checked for the matching keyword and the intent
    is returned.

    If the input sentence is pre-classified with an intent marked with keyword_intent
    metadata flag, the prior intent classification by an nlu classifier will be
    cleared.

    """

    defaults = {
        "case_sensitive": True,
        "match_condition": "contain",
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        intent_keyword_map: Optional[Dict] = None,
        is_keyword_intent_map: Optional[Dict] = None,
    ) -> None:

        super(KeywordIntentClassifier, self).__init__(component_config)

        self.case_sensitive = self.component_config.get("case_sensitive")
        self.match_condition = self.component_config.get("match_condition")
        self.intent_keyword_map = intent_keyword_map or {}
        self.is_keyword_intent_map = is_keyword_intent_map or {}

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        duplicate_examples = set()
        for ex in training_data.intent_examples:
            metadata = ex.get(METADATA, {})
            intent_metadata = metadata.get(METADATA_INTENT, {})
            example_metadata = metadata.get(METADATA_EXAMPLE, {})
            match_condition = example_metadata.get(
                "match_condition",
                intent_metadata.get("match_condition", self.match_condition),
            )
            keyword = (ex.get(TEXT), match_condition)

            is_keyword_intent = intent_metadata.get("keyword_intent", False)
            if ex.get(INTENT) not in self.is_keyword_intent_map.keys():
                self.is_keyword_intent_map[ex.get(INTENT)] = is_keyword_intent

            if (
                keyword in self.intent_keyword_map.keys()
                and ex.get(INTENT) != self.intent_keyword_map[keyword]
            ):
                duplicate_examples.add(keyword)
                rasa.shared.utils.io.raise_warning(
                    f"Keyword '{ex.get(TEXT)}' is a keyword to trigger intent "
                    f"'{self.intent_keyword_map[keyword]}' and also "
                    f"intent '{ex.get(INTENT)}', it will be removed "
                    f"from the list of keywords for both of them. "
                    f"Remove (one of) the duplicates from the training data.",
                    docs=DOCS_URL_COMPONENTS + "#keyword-intent-classifier",
                )
            else:
                self.intent_keyword_map[keyword] = ex.get(INTENT)
        for keyword in duplicate_examples:
            self.intent_keyword_map.pop(keyword)
            logger.debug(
                f"Removed '{keyword}' from the list of keywords because it was "
                "a keyword for more than one intent."
            )

        self._validate_keyword_map()

    def _validate_keyword_map(self) -> None:
        re_flag = 0 if self.case_sensitive else re.IGNORECASE

        ambiguous_mappings = []
        for keyword1, intent1 in self.intent_keyword_map.items():
            for keyword2, intent2 in self.intent_keyword_map.items():
                if (
                    keyword1[1] == keyword2[1]
                    and re.search(
                        r"\b" + keyword1[0] + r"\b", keyword2[0], flags=re_flag
                    )
                    and intent1 != intent2
                ):
                    ambiguous_mappings.append((intent1, keyword1))
                    rasa.shared.utils.io.raise_warning(
                        f"Keyword '{keyword1}' is a keyword of intent '{intent1}', "
                        f"but also a substring of '{keyword2}', which is a "
                        f"keyword of intent '{intent2}."
                        f" '{keyword1}' will be removed from the list of keywords.\n"
                        f"Remove (one of) the conflicting keywords from the"
                        f" training data.",
                        docs=DOCS_URL_COMPONENTS + "#keyword-intent-classifier",
                    )
        for intent, keyword in ambiguous_mappings:
            self.intent_keyword_map.pop(keyword)
            logger.debug(
                f"Removed keyword '{keyword}' from intent '{intent}' because it matched a "
                "keyword of another intent."
            )

    def process(self, message: Message, **kwargs: Any) -> None:
        intent = {"name": None, "confidence": 0.0}
        existing_intent = message.get(INTENT, {})
        existing_intent_name = existing_intent.get("name", "")
        if existing_intent is not None and self.is_keyword_intent_map.get(
            existing_intent_name, False
        ):
            logger.debug(
                f"Clearing intent classification as message '{message.get(TEXT)}' is classified with intent '{message.get(INTENT)}' and this intent is flagged as a keyword_intent"
            )
            message.set(INTENT, intent, add_to_output=True)

        intent_name = self._map_keyword_to_intent(message.get(TEXT))

        if intent_name is not None:
            intent = {"name": intent_name, "confidence": 1.0}

        if message.get(INTENT) is None or intent_name is not None:
            message.set(INTENT, intent, add_to_output=True)

    def _map_keyword_to_intent(self, text: Text) -> Optional[Text]:
        re_flag = 0 if self.case_sensitive else re.IGNORECASE

        matched_intent = None
        matched_condition = None
        for keyword, intent in self.intent_keyword_map.items():
            keyword_text = keyword[0]
            keyword_match_condition = keyword[1]

            match_condition_regex = self._get_regex_for_match_condition(
                keyword_text, keyword_match_condition
            )
            if match_condition_regex is None:
                continue

            if re.search(match_condition_regex, text, flags=re_flag):
                logger.debug(
                    f"KeywordClassifier matched keyword '{keyword}' to"
                    f" intent '{intent}'."
                )
                if (matched_intent is None) or (
                    matched_intent is not None
                    and self._get_priority_for_match_condition(keyword_match_condition)
                    > self._get_priority_for_match_condition(matched_condition)
                ):
                    matched_intent = intent
                    matched_condition = keyword_match_condition

        return matched_intent

    def _get_regex_for_match_condition(self, keyword_text: Text, match_condition: Text):
        match_condition_regex_map = {
            "contain": r"(^|(?<=\s))" + re.escape(keyword_text) + r"($|(?=\s))",
            "exact": r"^" + re.escape(keyword_text) + r"$",
            "start": r"^" + re.escape(keyword_text) + r"($|(?=\s))",
            "end": r"(^|(?<=\s))" + re.escape(keyword_text) + r"$",
        }
        return match_condition_regex_map.get(match_condition, None)

    def _get_priority_for_match_condition(self, match_condition: Text) -> int:
        match_condition_priority = {
            "contain": 1,
            "end": 2,
            "start": 3,
            "exact": 4,
        }
        return match_condition_priority.get(match_condition, 0)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        file_name = file_name + ".json"
        keyword_file = os.path.join(model_dir, file_name)
        intent_keyword_map = {
            (k[0] + k[1]): v for k, v in self.intent_keyword_map.items()
        }
        keyword_file_data = {
            "intent_keyword_map": intent_keyword_map,
            "is_keyword_intent_map": self.is_keyword_intent_map,
        }
        utils.write_json_to_file(keyword_file, keyword_file_data)

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: Metadata = None,
        cached_component: Optional["KeywordIntentClassifier"] = None,
        **kwargs: Any,
    ) -> "KeywordIntentClassifier":
        """Loads trained component (see parent class for full docstring)."""
        if meta.get("file"):
            file_name = meta.get("file")
            keyword_file = os.path.join(model_dir, file_name)
            if os.path.exists(keyword_file):
                keyword_file_data = rasa.shared.utils.io.read_json_file(keyword_file)
                intent_keyword_map = keyword_file_data.get("intent_keyword_map", None)
                intent_keyword_map = {
                    _get_keyword_tuple(k): v for k, v in intent_keyword_map.items()
                }
                is_keyword_intent_map = keyword_file_data.get(
                    "is_keyword_intent_map", None
                )
            else:
                rasa.shared.utils.io.raise_warning(
                    f"Failed to load key word file for `IntentKeywordClassifier`, "
                    f"maybe {keyword_file} does not exist?"
                )
                intent_keyword_map = None
                is_keyword_intent_map = None
            return cls(meta, intent_keyword_map, is_keyword_intent_map)
        else:
            raise Exception(
                f"Failed to load keyword intent classifier model. "
                f"Path {os.path.abspath(meta.get('file'))} doesn't exist."
            )


def _get_keyword_tuple(key):
    match = re.search(r"(contain)$|(exact)$|(start)$|(end)$", key)
    if match is None:
        logger.debug("match_condition not found in model file, defaulting to contain")
        return (key, "contain")
    match_condition = match.group()
    keyword_text = key[: 0 - len(match_condition)]
    return (keyword_text, match_condition)
