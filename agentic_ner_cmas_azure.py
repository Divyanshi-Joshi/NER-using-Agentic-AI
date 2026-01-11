"""
Architecture:
1. Self-Annotator Agent: Generates self-annotated data from unlabeled corpus
2. TRF Extractor Agent: Identifies Type-Related Features (contextual correlations)
3. Demonstration Discriminator Agent: Evaluates helpfulness of demonstrations
4. Overall Predictor Agent: Final NER predictions with self-consistency
"""

import os
import json
import re
from typing import TypedDict, Annotated, Any
from collections import defaultdict
import math
from functools import reduce
import operator

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# LangChain imports - MODIFIED FOR AZURE
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import secure configuration from config.py
from config import get_azure_config, verify_azure_config

# ==================== DATA MODELS ====================

class EntityPrediction(BaseModel):
    """Model for entity predictions"""
    entity: str = Field(description="The entity text")
    entity_type: str = Field(description="The entity type/label")
    confidence: float = Field(description="Confidence score")

class TRFPrediction(BaseModel):
    """Model for Type-Related Features"""
    trf: str = Field(description="Type-related feature token")
    entity_type: str = Field(description="Associated entity type")
    relevance_score: float = Field(description="Relevance score")

class HelpfulnessScore(BaseModel):
    """Model for demonstration helpfulness scoring"""
    demonstration_id: int = Field(description="ID of the demonstration")
    helpfulness_score: float = Field(description="Helpfulness score (1-5)")
    reasoning: str = Field(description="Reasoning for the score")

class CMASState(TypedDict):
    """State structure for CMAS workflow"""
    # Input
    target_sentence: str
    entity_types: list[str]
    unlabeled_corpus: list[str]
    
    # Self-Annotator outputs
    self_annotated_data: dict  # {sentence: [entity predictions]}
    selected_demonstrations: list[dict]  # Retrieved demonstration sentences
    
    # TRF Extractor outputs
    trf_set: dict  # {entity_type: [TRF tokens]}
    demonstration_trfs: dict  # {demo_id: {entity_type: [TRFs]}}
    target_sentence_trfs: list[str]
    
    # Demonstration Discriminator outputs
    helpfulness_scores: dict  # {demo_id: score}
    filtered_demonstrations: list[dict]
    
    # Overall Predictor outputs
    final_predictions: list[EntityPrediction]
    confidence_scores: list[float]
    
    # Metadata
    iteration: int
    errors: list[str]

# ==================== AGENT IMPLEMENTATIONS ====================

class SelfAnnotatorAgent:
    """
    Generates self-annotated predictions on unlabeled corpus.
    Uses self-consistency voting for reliability.
    MODIFIED: Uses Azure OpenAI
    """
    
    def __init__(self, temperature: float = 0.7):
        # MODIFICATION #1: Use AzureChatOpenAI instead of ChatOpenAI
        azure_config = get_azure_config()
        self.llm = AzureChatOpenAI(
            model="o4-mini",
            deployment_name=azure_config["deployment_name"],
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"],
            azure_endpoint=azure_config["azure_endpoint"]
        )
        self.parser = JsonOutputParser(pydantic_object=list[EntityPrediction])
    
    def generate_self_annotations(self, sentence: str, entity_types: list[str], num_samples: int = 5) -> list[EntityPrediction]:
        """
        Generate multiple predictions using self-consistency voting.
        Returns: Most voted entity predictions
        """
        
        entity_types_str = ", ".join(entity_types)
        
        prompt = f"""You are an expert NER, PII, and PCI detection system. Extract all named entities from the text, paying special attention to sensitive and obfuscated data.

        Entity types to recognize: {entity_types_str}

        Definitions:
        - PHONE_NUMBER: Extract digits (555-0199) AND text-based numbers (e.g., "nine eight one...").
        - EMAIL: Extract standard emails and obfuscated ones (e.g., "user(at)domain..com", "user@@domain").
        - CREDIT_CARD: Extract 16-digit numbers with spaces or dashes.
        - DISEASE/MEDICATION: Extract medical conditions and drug names.

        Text: "{sentence}"

        Return a JSON list of entities with this format:
        [{{"entity": "entity_text", "entity_type": "TYPE", "confidence": 0.95}}]

        If no entities found, return empty list: []
        """
        
        predictions_list = []
        
        # Self-consistency: Generate multiple samples
        for i in range(num_samples):
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                response_text = response.content
                
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    predictions = json.loads(json_str)
                    predictions_list.append(predictions)
            except Exception as e:
                continue
        
        # Self-consistency voting: Keep entities appearing in >50% of samples
        if not predictions_list:
            return []
        
        entity_votes = defaultdict(lambda: {"count": 0, "types": defaultdict(int)})
        
        for pred_list in predictions_list:
            for pred in pred_list:
                entity_text = pred.get("entity", "").lower()
                entity_type = pred.get("entity_type", "")
                entity_votes[entity_text]["count"] += 1
                entity_votes[entity_text]["types"][entity_type] += 1
        
        threshold = num_samples / 2
        voted_predictions = []
        
        for entity_text, vote_info in entity_votes.items():
            if vote_info["count"] >= threshold:
                # Get most voted type
                most_voted_type = max(vote_info["types"].items(), key=operator.itemgetter(1))[0]
                confidence = vote_info["count"] / num_samples
                voted_predictions.append(EntityPrediction(
                    entity=entity_text,
                    entity_type=most_voted_type,
                    confidence=confidence
                ))
        
        return voted_predictions
    
    def annotate_corpus(self, corpus: list[str], entity_types: list[str]) -> dict:
        """Annotate entire unlabeled corpus"""
        annotations = {}
        
        for i, sentence in enumerate(corpus):
            print(f"  Annotating corpus sentence {i+1}/{len(corpus)}...")
            predictions = self.generate_self_annotations(sentence, entity_types, num_samples=3)
            
            # Convert to dicts with proper error handling
            entities_list = []
            for pred in predictions:
                try:
                    if isinstance(pred, dict):
                        entities_list.append(pred)
                    elif hasattr(pred, 'model_dump'):
                        entities_list.append(pred.model_dump())
                    else:
                        entities_list.append({
                            "entity": str(getattr(pred, "entity", "")),
                            "entity_type": str(getattr(pred, "entity_type", "")),
                            "confidence": float(getattr(pred, "confidence", 0.5))
                        })
                except Exception as e:
                    # Fallback
                    entities_list.append({
                        "entity": str(getattr(pred, "entity", "")),
                        "entity_type": str(getattr(pred, "entity_type", "")),
                        "confidence": 0.5
                    })
            
            annotations[sentence] = entities_list
        
        return annotations

    
    def retrieve_demonstrations(self, target_sentence: str, annotations: dict, k: int = 5) -> list[dict]:
        """Retrieve k most similar demonstrations using semantic similarity."""
        import random
        
        try:
            azure_config = get_azure_config()
            
            # Use EMBEDDING-specific API version
            embeddings = AzureOpenAIEmbeddings(
                model=azure_config["embedding_deployment_name"],
                api_key=azure_config["api_key"],
                api_version=azure_config["embedding_api_version"],  
                azure_endpoint=azure_config["azure_endpoint"]
            )
            
            target_embedding = embeddings.embed_query(target_sentence)
            
            corpus_sentences = list(annotations.keys())
            
            if not corpus_sentences:
                return []
            
            corpus_embeddings = embeddings.embed_documents(corpus_sentences)
            
            # Step 1: Get 1D array
            similarities_2d = cosine_similarity([target_embedding], corpus_embeddings)
            similarities = similarities_2d[0]  # ← Extract first row

            # Step 2: Sort correctly
            sorted_indices = np.argsort(similarities)
            top_indices = sorted_indices[-k:]
            top_indices = top_indices[::-1]

            # Step 3: Convert numpy int to Python int
            demonstrations = []
            for idx in top_indices:
                idx_int = int(idx)  # ← Convert numpy int64 to Python int
                sentence = corpus_sentences[idx_int]
                demonstrations.append({
                    "sentence": sentence,
                    "entities": annotations[sentence],
                    "similarity": float(similarities[idx_int])
                })

            
            return demonstrations
            
        except Exception as e:
            print(f"Embeddings error: {e}, using fallback...")
            corpus_sentences = list(annotations.keys())
            selected_count = min(k, len(corpus_sentences))
            
            if selected_count == 0:
                return []
            
            selected = random.sample(corpus_sentences, selected_count)
            
            return [
                {
                    "sentence": s,
                    "entities": annotations[s],
                    "similarity": 0.5
                }
                for s in selected
            ]


class TRFExtractorAgent:
    """
    Extracts Type-Related Features (contextual correlations) for entities.
    Identifies tokens strongly associated with entity types.
    MODIFIED: Uses Azure OpenAI
    """
    
    def __init__(self):
        # MODIFICATION #3: Use AzureChatOpenAI
        azure_config = get_azure_config()
        self.llm = AzureChatOpenAI(
            model="o4-mini",
            deployment_name=azure_config["deployment_name"],
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"],
            azure_endpoint=azure_config["azure_endpoint"]
        )
    
    def compute_mutual_information(self, corpus: list[str], annotations: dict, entity_types: list[str], rho: float = 3.0) -> dict:
        """
        Compute Type-Related Features using mutual information criterion.
        
        For each entity type t:
        - D_t: sentences containing entities of type t
        - D_u \ D_t: sentences not containing type t
        - Include word w if: count(w in D_t) / count(w in D_u \ D_t) >= rho
        """
        
        # Build D_t for each entity type
        entity_type_sentences = defaultdict(list)
        
        for sentence, entities in annotations.items():
            for entity in entities:
                entity_type = entity.get("entity_type", "")
                if entity_type:
                    entity_type_sentences[entity_type].append(sentence)
        
        # Count word frequencies
        trf_set = {}
        
        for entity_type in entity_types:
            D_t = entity_type_sentences.get(entity_type, [])
            D_complement = [s for s in corpus if s not in D_t]
            
            if not D_t:
                trf_set[entity_type] = []
                continue
            
            # Extract 1-grams from D_t
            words_in_Dt = defaultdict(int)
            words_in_complement = defaultdict(int)
            
            for sentence in D_t:
                words = sentence.lower().split()
                for word in words:
                    words_in_Dt[word] += 1
            
            for sentence in D_complement:
                words = sentence.lower().split()
                for word in words:
                    words_in_complement[word] += 1
            
            # Apply mutual information filter
            trfs = []
            for word, count_Dt in words_in_Dt.items():
                count_complement = words_in_complement.get(word, 0)
                
                # Avoid division by zero
                if count_complement == 0:
                    ratio = float('inf')
                else:
                    ratio = count_Dt / (count_complement + 1e-6)
                
                # Apply threshold
                if count_Dt > 0 and ratio >= rho:
                    trfs.append(word)
            
            trf_set[entity_type] = trfs[:20]  # Top 20 TRFs per type
        
        return trf_set
    
    def extract_trfs_from_sentence(self, sentence: str, entity_types: list[str], trf_set: dict) -> list[str]:
        """
        Extract TRFs present in the target sentence for its entity types.
        """
        sentence_words = set(sentence.lower().split())
        
        extracted_trfs = []
        for entity_type in entity_types:
            type_trfs = trf_set.get(entity_type, [])
            for trf in type_trfs:
                if trf in sentence_words:
                    extracted_trfs.append(trf)
        
        return list(set(extracted_trfs))  # Remove duplicates
    
    def generate_trf_labels(self, demonstrations: list[dict], sentence: str, entity_types: list[str], trf_set: dict) -> dict:
        """
        Generate TRF labels for demonstrations using ICL (In-Context Learning).
        """
        
        prompt = f"""You are an expert at identifying contextual features for named entities.

Entity types: {", ".join(entity_types)}

Type-Related Features (TRFs) are words/phrases strongly associated with entity types.

Examples:
- For Person entities: "famous", "actor", "scientist", "president"
- For Organization: "company", "founded", "CEO", "Inc"
- For Location: "city", "country", "located", "capital"

Sentence: "{sentence}"

List the TRFs relevant to {entity_types} that appear in this sentence.
Return as JSON: ["trf1", "trf2", ...]
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content
            
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                trfs = json.loads(json_match.group())
                return {"extracted_trfs": trfs}
        except Exception as e:
            pass
        
        # Fallback to MI-based extraction
        return {"extracted_trfs": self.extract_trfs_from_sentence(sentence, entity_types, trf_set)}

class DemonstrationDiscriminatorAgent:
    """
    Evaluates helpfulness of retrieved demonstrations.
    Incorporates self-reflection mechanism to filter irrelevant examples.
    MODIFIED: Uses Azure OpenAI
    """
    
    def __init__(self):
        # MODIFICATION #4: Use AzureChatOpenAI
        azure_config = get_azure_config()
        self.llm = AzureChatOpenAI(
            model="o4-mini",
            deployment_name=azure_config["deployment_name"],
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"],
            azure_endpoint=azure_config["azure_endpoint"]
        )
    
    def evaluate_demonstration_helpfulness(self, 
                                          demonstration: dict,
                                          target_sentence: str,
                                          target_trfs: list[str],
                                          entity_types: list[str]) -> float:
        """
        Score how helpful a demonstration is for predicting entities in target sentence.
        
        Factors:
        - TRF overlap between demo and target
        - Entity type coverage in demo
        - Syntactic similarity
        """
        
        demo_sentence = demonstration.get("sentence", "")
        demo_entities = demonstration.get("entities", [])
        demo_trfs = demonstration.get("extracted_trfs", [])
        
        # Factor 1: TRF overlap
        target_trf_set = set(target_trfs)
        demo_trf_set = set(demo_trfs)
        trf_overlap = len(target_trf_set & demo_trf_set) / (len(target_trf_set | demo_trf_set) + 1e-6)
        
        # Factor 2: Entity type coverage
        demo_types = set([e.get("entity_type", "") for e in demo_entities])
        target_type_set = set(entity_types)
        type_coverage = len(demo_types & target_type_set) / (len(target_type_set) + 1e-6)
        
        # Factor 3: Length similarity (longer demos often more informative)
        demo_length = len(demo_sentence.split())
        target_length = len(target_sentence.split())
        length_similarity = 1 - abs(demo_length - target_length) / max(demo_length, target_length, 1)
        
        # Combined score (weights can be tuned)
        helpfulness = (0.4 * trf_overlap) + (0.4 * type_coverage) + (0.2 * length_similarity)
        
        return min(5.0, helpfulness * 5)  # Scale to 1-5
    
    def score_demonstrations(self, 
                            demonstrations: list[dict],
                            target_sentence: str,
                            target_trfs: list[str],
                            entity_types: list[str]) -> dict:
        """
        Score all demonstrations and return with scores.
        """
        
        scored_demos = []
        
        for idx, demo in enumerate(demonstrations):
            score = self.evaluate_demonstration_helpfulness(
                demo, target_sentence, target_trfs, entity_types
            )
            
            demo_with_score = {**demo, "demo_id": idx, "helpfulness_score": score}
            scored_demos.append(demo_with_score)
        
        return {"scored_demonstrations": scored_demos}
    
    def filter_by_threshold(self, scored_demos: list[dict], threshold: float = 2.5) -> list[dict]:
        """Filter out low-scoring demonstrations"""
        return [d for d in scored_demos if d.get("helpfulness_score", 0) >= threshold]

class OverallPredictorAgent:
    """
    Final prediction agent that uses TRFs and demonstration scores.
    Employs two-stage self-consistency for reliable predictions.
    MODIFIED: Uses Azure OpenAI
    """
    
    def __init__(self):
        # MODIFICATION #5: Use AzureChatOpenAI
        azure_config = get_azure_config()
        self.llm = AzureChatOpenAI(
            model="o4-mini",
            deployment_name=azure_config["deployment_name"],
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"],
            azure_endpoint=azure_config["azure_endpoint"]
        )
    
    def predict_entities(self, 
                        target_sentence: str,
                        entity_types: list[str],
                        scored_demonstrations: list[dict],
                        target_trfs: list[str],
                        num_samples: int = 5) -> list[EntityPrediction]:
        """
        Predict entities using demonstrations, TRFs, and self-consistency.
        """
        
        # Build demonstration context
        demo_context = ""
        for demo in scored_demonstrations[:3]:  # Top 3 most helpful
            score = demo.get("helpfulness_score", 0)
            entities = demo.get("entities", [])
            trfs = demo.get("extracted_trfs", [])
            
            entity_str = ", ".join([f"{e['entity']} ({e['entity_type']})" for e in entities])
            
            demo_context += f"""
Example (Helpfulness: {score:.1f}/5):
Text: "{demo['sentence']}"
TRFs: {", ".join(trfs[:5])}
Entities: {entity_str}
---
"""
        
        entity_types_str = ", ".join(entity_types)
        
        prompt = f"""You are an expert NER system.

Entity types: {entity_types_str}
Target entity-related features: {", ".join(target_trfs[:5])}

{demo_context}

Now extract entities from:
Text: "{target_sentence}"

Consider the entity-related features and demonstrations above.
Return JSON: [{{"entity": "text", "entity_type": "TYPE", "confidence": 0.95}}]
Return [] if no entities found.
"""
        
        predictions_list = []
        
        # Self-consistency: Multiple samples
        for i in range(num_samples):
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                response_text = response.content
                
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    predictions = json.loads(json_str)
                    predictions_list.append(predictions)
            except Exception as e:
                print(f"!! ERROR in generate_self_annotations: {e}")
                continue
        
        if not predictions_list:
            return []
        
        # Two-stage voting for reliability
        return self._two_stage_voting(predictions_list)
    
    def _two_stage_voting(self, predictions_list: list) -> list[EntityPrediction]:
        """
        Two-stage voting:
        Stage 1: Keep mention if appears in >50% of samples
        Stage 2: Assign type by majority vote
        """
        
        entity_votes = defaultdict(lambda: {"count": 0, "types": defaultdict(int)})
        
        for pred_list in predictions_list:
            for pred in pred_list:
                entity_text = pred.get("entity", "").lower()
                entity_type = pred.get("entity_type", "")
                confidence = pred.get("confidence", 0.5)
                
                entity_votes[entity_text]["count"] += 1
                entity_votes[entity_text]["types"][entity_type] += 1
                entity_votes[entity_text]["confidences"] = entity_votes[entity_text].get("confidences", []) + [confidence]
        
        threshold = len(predictions_list) / 2
        final_predictions = []
        
        for entity_text, vote_info in entity_votes.items():
            if vote_info["count"] >= threshold:
                # Stage 2: Get most voted type
                most_voted_type = max(vote_info["types"].items(), key=operator.itemgetter(1))[0]
                avg_confidence = np.mean(vote_info.get("confidences", [0.5]))
                
                final_predictions.append(EntityPrediction(
                    entity=entity_text,
                    entity_type=most_voted_type,
                    confidence=avg_confidence
                ))
        
        return final_predictions

# ==================== LANGGRAPH WORKFLOW ====================

def create_cmas_workflow():
    """
    Create LangGraph workflow implementing CMAS.
    """
    
    workflow = StateGraph(CMASState)
    
    # Initialize agents
    self_annotator = SelfAnnotatorAgent()
    trf_extractor = TRFExtractorAgent()
    discriminator = DemonstrationDiscriminatorAgent()
    predictor = OverallPredictorAgent()
    
    # Node 1: Self-Annotation
    def self_annotation_node(state: CMASState) -> Command:
        """Step 1: Generate self-annotated data and retrieve demonstrations"""
        
        try:
            # Annotate corpus
            annotations = self_annotator.annotate_corpus(
                state["unlabeled_corpus"],
                state["entity_types"]
            )
            
            # Retrieve demonstrations
            demonstrations = self_annotator.retrieve_demonstrations(
                state["target_sentence"],
                annotations,
                k=5
            )
            
            return Command(
                update={
                    "self_annotated_data": annotations,
                    "selected_demonstrations": demonstrations,
                    "iteration": state.get("iteration", 0) + 1
                },
                goto="trf_extraction"
            )
        except Exception as e:
            return Command(
                update={"errors": state.get("errors", []) + [f"Self-annotation error: {str(e)}"]},
                goto=END
            )
    
    # Node 2: TRF Extraction
    def trf_extraction_node(state: CMASState) -> Command:
        """Step 2: Extract type-related features"""
        
        try:
            # Compute mutual information TRF set
            trf_set = trf_extractor.compute_mutual_information(
                state["unlabeled_corpus"],
                state["self_annotated_data"],
                state["entity_types"],
                rho=3.0
            )
            
            # Extract TRFs for target sentence
            target_trfs = trf_extractor.extract_trfs_from_sentence(
                state["target_sentence"],
                state["entity_types"],
                trf_set
            )
            
            # Generate TRF labels for demonstrations
            demo_trfs = {}
            for demo in state["selected_demonstrations"]:
                demo_trfs[demo["sentence"]] = trf_extractor.extract_trfs_from_sentence(
                    demo["sentence"],
                    state["entity_types"],
                    trf_set
                )
            
            return Command(
                update={
                    "trf_set": trf_set,
                    "target_sentence_trfs": target_trfs,
                    "demonstration_trfs": demo_trfs
                },
                goto="demonstration_discrimination"
            )
        except Exception as e:
            return Command(
                update={"errors": state.get("errors", []) + [f"TRF extraction error: {str(e)}"]},
                goto=END
            )
    
    # Node 3: Demonstration Discrimination
    def demonstration_discrimination_node(state: CMASState) -> Command:
        """Step 3: Score demonstration helpfulness"""
        
        try:
            # Add TRFs to demonstrations
            for demo in state["selected_demonstrations"]:
                demo_sentence = demo["sentence"]
                demo["extracted_trfs"] = state["demonstration_trfs"].get(demo_sentence, [])
            
            # Score demonstrations
            scored = discriminator.score_demonstrations(
                state["selected_demonstrations"],
                state["target_sentence"],
                state["target_sentence_trfs"],
                state["entity_types"]
            )
            
            scored_demos = scored["scored_demonstrations"]
            
            # Filter by threshold
            filtered_demos = discriminator.filter_by_threshold(scored_demos, threshold=2.5)
            
            return Command(
                update={
                    "helpfulness_scores": {d["demo_id"]: d["helpfulness_score"] for d in scored_demos},
                    "filtered_demonstrations": filtered_demos if filtered_demos else scored_demos[:3]
                },
                goto="overall_prediction"
            )
        except Exception as e:
            return Command(
                update={"errors": state.get("errors", []) + [f"Discrimination error: {str(e)}"]},
                goto=END
            )
    
    # Node 4: Overall Prediction
    def overall_prediction_node(state: CMASState) -> Command:
        """Step 4: Generate final NER predictions"""
        
        try:
            final_predictions = predictor.predict_entities(
                state["target_sentence"],
                state["entity_types"],
                state["filtered_demonstrations"],
                state["target_sentence_trfs"],
                num_samples=5
            )
            
            confidence_scores = [p.confidence for p in final_predictions]
            
            return Command(
                update={
                    "final_predictions": final_predictions,
                    "confidence_scores": confidence_scores
                },
                goto=END
            )
        except Exception as e:
            return Command(
                update={"errors": state.get("errors", []) + [f"Prediction error: {str(e)}"]},
                goto=END
            )
    
    # Add nodes to workflow
    workflow.add_node("self_annotation", self_annotation_node)
    workflow.add_node("trf_extraction", trf_extraction_node)
    workflow.add_node("demonstration_discrimination", demonstration_discrimination_node)
    workflow.add_node("overall_prediction", overall_prediction_node)
    
    # Define edges
    workflow.add_edge(START, "self_annotation")
    
    # Compile workflow
    return workflow.compile()

# ==================== MAIN EXECUTION ====================

def run_cmas_ner(target_sentence: str, 
                 entity_types: list[str],
                 unlabeled_corpus: list[str]) -> dict:
    """
    Run CMAS workflow for NER on a target sentence.
    
    Args:
        target_sentence: Sentence to extract entities from
        entity_types: List of entity types to recognize
        unlabeled_corpus: Unlabeled corpus for self-annotation
    
    Returns:
        Dictionary with predictions and metadata
    """
    
    # Verify Azure configuration
    verify_azure_config()
    
    # Create workflow
    workflow = create_cmas_workflow()
    
    # Initial state
    initial_state = CMASState(
        target_sentence=target_sentence,
        entity_types=entity_types,
        unlabeled_corpus=unlabeled_corpus,
        self_annotated_data={},
        selected_demonstrations=[],
        trf_set={},
        demonstration_trfs={},
        target_sentence_trfs=[],
        helpfulness_scores={},
        filtered_demonstrations=[],
        final_predictions=[],
        confidence_scores=[],
        iteration=0,
        errors=[]
    )
    
    # Execute workflow
    result = workflow.invoke(initial_state)
    
    return {
        "target_sentence": result["target_sentence"],
        "entity_types": result["entity_types"],
        "predictions": [
            {
                "entity": p.entity,
                "entity_type": p.entity_type,
                "confidence": p.confidence
            }
            for p in result["final_predictions"]
        ],
        "trfs_identified": result["target_sentence_trfs"],
        "num_demonstrations_used": len(result["filtered_demonstrations"]),
        "helpfulness_scores": result["helpfulness_scores"],
        "errors": result["errors"],
        "num_iterations": result["iteration"]
    }

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example corpus
    unlabeled_corpus = [
    "Dr. Amit Patel prescribed Amoxicillin for the patient's severe throat infection.",
    "Patient ID 8821-A reported recurring migraines and nausea after taking Ibuprofen.",
    "The claim sent to claims@starhealth.in was rejected due to a missing signature.",
    "Please verify the transaction on card 4455-6677-8899-0011 with CVV 456.",
    "Emergency contact is available at nine one one two three four five six seven eight.",
    "Card ending in 1122 expires on 12/25 and was used for the copay at Apollo Hospital.",
    "The email address provided, support(at)medlife.com, seems to be invalid.",
    "Diagnosed with Stage 1 Hypertension, the patient was advised to start Lisinopril.",
    "Refund of $200 processed to card 5123 4567 8901 2345.",
    "Contact the billing department at billing.dept@fortis_hospital.org for invoice #992.",
    "Mr. John Doe's insulin prescription was renewed by Dr. Sarah Lee.",
    "Payment declined for CVV 998 on the corporate AMEX card.",
    "The patient admitted to City Care Clinic has a history of Type 1 Diabetes.",
    "Submit the reimbursement form to reimbursement-help@@insurance..co..in immediately.",
    "Card expiry 09/30 matches the records for Patient P-112233.",
    "Prescribed Atorvastatin 10mg daily for high cholesterol management.",
    "The insurance policy covers treatments at Max Healthcare and associated labs.",
    "Please update the contact info to eight eight zero zero one one two two three three.",
    "Unauthorized charge detected on 4111-1111-1111-1111.",
    "Dr. Rao recommended an MRI scan at NeuroScans Diagnostics.",
    "The patient is allergic to Penicillin and Sulfa drugs.",
    "Send the report to lab_results@pathkind.labs or the backup email.",
    "Invalid CVV 000 entered for the transaction on 15th April.",
    "Aarogya Setu app requires linking the mobile number seven nine nine nine...",
    "Patient complains of chronic back pain and was given Tramadol.",
    "The settlement was transferred to account linked with card 4000 1234 5678 9010.",
    "Verify insurance eligibility for P-55443 with BlueCross BlueShield.",
    "Email correspondence sent to grievances@policybazaar.com was acknowledged.",
    "Dr. Neha Gupta treated the fracture at St. Jude's Medical Center.",
    "The pharmacy at 123 Health St accepts cards with expiry up to 12/30.",
    "Switching medication from Aspirin to Clopidogrel due to side effects.",
    "The user claimed the email id john.doe@@gmail..com is typo-prone.",
    "Confirm payment of INR 5000 using the card 6011-0000-0000-0000.",
    "Patient exhibited signs of acute bronchitis and mild fever.",
    "Contact Dr. Who at five five five one two three four.",
    "Insurance claim #998877 was denied by UnitedHealth Group.",
    "Sensitive data like CVV 111 should not be stored in plain text.",
    "The patient was referred to a cardiologist at HeartCare Institute.",
    "Update the record for Mrs. Smith, DOB 01/01/1980.",
    "Payment gateway rejected the card due to incorrect expiry 01/20."
    ]
    
    # Test cases
    test_cases = [
        {
        "sentence": """On 12 March 2025, Dr. Kavita Sen from Lotus Heart Clinic reviewed the medical file of Patient ID: P-98123, who reported symptoms of hypertension, Type-2 diabetes, and mild asthma. She suggested switching from Metformin to Glyxora-XR manufactured by Medivance Biotech Pvt. Ltd.

        The patient mentioned they recently bought a health plan from AarogyaShield Insurance, but their previous provider, Universal Care Corp, refused reimbursement because the claim form was sent from the wrong email rahul.health@@gmail..com instead of the correct rahul.health.support@universalcare.com
        . The internal department also logged a complaint sent to claims-dept@aarogyashield.in
        , although another version of the email appeared corrupted as: claims-dept(at)aarogya_shield..in.

        During admission, the patient provided these payment details:
        Card Number: 4321 5678 9012 3456
        Expiry: 08/29
        CVV: 123

        Their alternate card was:
        Card Number: 5532-1122-9900-8844
        CVV: 981

        The patient’s emergency contact, Mrs. Shalini Rao, reachable at nine eight one two three four four one zero nine, works at Novacura Labs, a company collaborating with HealSync AI, an American health-tech firm known for its HIPAA-compliant analytics platform.""",
                "entity_types": [
                    "PERSON", "ORGANIZATION", "EMAIL", "PHONE_NUMBER", 
                    "CREDIT_CARD", "CVV", "EXPIRY_DATE", 
                    "DISEASE", "MEDICATION", "MEDICAL_ID", "DATE"
                ]
        }
    ]
    
    print("=" * 80)
    print("CMAS NER System - Azure OpenAI Implementation")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Sentence: {test['sentence']}")
        print(f"Entity Types: {test['entity_types']}")
        print("-" * 80)
        
        try:
            result = run_cmas_ner(
                target_sentence=test['sentence'],
                entity_types=test['entity_types'],
                unlabeled_corpus=unlabeled_corpus
            )
            
            print("\\n[RESULTS]")
            print(f"Identified TRFs: {result['trfs_identified']}")
            print(f"Demonstrations Used: {result['num_demonstrations_used']}")
            print(f"\\nPredicted Entities:")
            
            if result['predictions']:
                for pred in result['predictions']:
                    print(f"  - {pred['entity']:20} ({pred['entity_type']:15}) [Confidence: {pred['confidence']:.2f}]")
            else:
                print("  No entities found")
            
            if result['errors']:
                print(f"\\nErrors: {result['errors']}")
        
        except Exception as e:
            print(f"Error processing test case: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\\n" + "=" * 80)
