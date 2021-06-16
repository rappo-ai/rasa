---
sidebar_label: rasa.nlu.classifiers.keyword_intent_classifier
title: rasa.nlu.classifiers.keyword_intent_classifier
---
## KeywordIntentClassifier Objects

```python
class KeywordIntentClassifier(IntentClassifier)
```

Intent classifier looking for keyword matches in the training examples
with the ability to customise the match condition, and override an existing
classification if a keyword intent is misclassified by any prior NLU classifier.

The classifier takes a list of keywords, match conditions and associated intents
as an input. An input sentence is checked for the matching keyword and the intent
is returned.

If the input sentence is pre-classified with an intent marked with keyword_intent
metadata flag, the prior intent classification by an nlu classifier will be
cleared.

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Dict[Text, Any]
```

Persist this model into the passed directory.

Return the metadata necessary to load the model again.

#### load

```python
 | @classmethod
 | load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Metadata = None, cached_component: Optional["KeywordIntentClassifier"] = None, **kwargs: Any, ,) -> "KeywordIntentClassifier"
```

Loads trained component (see parent class for full docstring).

