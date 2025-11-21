"""Utility functions."""
import os
import logging
import argparse
import pickle

import wandb

from evaluate import load

from models.huggingface_models import HuggingfaceModel
from utils import openai as oai

BRIEF_PROMPTS = {
    'default': "Answer the following question in a complete sentence.\n",
    'chat': 'Answer the following question in a single complete sentence.\n',
    'sentence': 'Answer the following question in a complete, informative sentence.\n',
    'detailed': 'Provide a detailed, well-structured answer to the following question. Write your response as a complete sentence that thoroughly addresses the question.\n',
    'short': 'Answer the following question with a brief, concise answer. Use only a few words or a short phrase, not a full sentence.\n',
    'manual': """Answer the following question in one complete sentence.

Context: The Battle of Waterloo was fought on Sunday, 18 June 1815 near Waterloo in present-day Belgium. A French army under Napoleon Bonaparte was defeated by two armies of the Seventh Coalition, including a British-led allied army under the Duke of Wellington.
Question: Who commanded the British forces at the Battle of Waterloo?
Answer: The Duke of Wellington commanded the British-led allied forces at the Battle of Waterloo in 1815.

Context: The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. Considered an archetypal masterpiece of the Italian Renaissance, it has been described as the best known, most visited, most written about, and most parodied work of art in the world.
Question: Who painted the Mona Lisa?
Answer: Leonardo da Vinci painted the Mona Lisa, which is considered a masterpiece of the Italian Renaissance.

Context: Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. Its elevation of 8,848.86 m was most recently established in 2020 by the Chinese and Nepali authorities.
Question: What is the height of Mount Everest?
Answer: Mount Everest stands at an elevation of 8,848.86 meters above sea level, making it Earth's highest mountain.

"""
}


def get_parser(stages=['generate', 'compute']):
    entity = os.getenv('WANDB_SEM_UNC_ENTITY', None)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=False,
        help="Keep default wandb clean.")
    parser.add_argument('--entity', type=str, default=entity)
    parser.add_argument('--project', type=str, default=None,
                        help='WandB project name (defaults to semantic_uncertainty if not set)')
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument(
        "--metric", type=str, default="squad",
        choices=['squad', 'llm', 'llm_gpt-3.5', 'llm_gpt-4', 'llm_llama-3.1-70b', 'llm_llama-3-70b', 'llm_llama-3-8b', 'llm_llama-3.1-8b', 'llm_llama-2-70b', 'llm_llama-2-13b'],
        help="Metric to assign accuracy to generations. Supports OpenAI models (llm_gpt-*) and HuggingFace Llama models (llm_llama-*).")
    parser.add_argument(
        "--compute_accuracy_at_all_temps",
        action=argparse.BooleanOptionalAction, default=True,
        help="Compute accuracy at all temperatures or only t<<1.")
    parser.add_argument(
        "--experiment_lot", type=str, default='Unnamed Experiment',
        help="Keep default wandb clean.")
    if 'generate' in stages:
        parser.add_argument(
            "--model_name", type=str, default="Llama-2-7b-chat", help="Model name",
        )
        parser.add_argument(
            "--model_max_new_tokens", type=int, default=150,
            help="Max number of tokens generated.",
        )
        parser.add_argument(
            "--dataset", type=str, default="trivia_qa",
            choices=['trivia_qa', 'squad', 'bioasq', 'nq', 'svamp'],
            help="Dataset to use")
        parser.add_argument(
            "--ood_train_dataset", type=str, default=None,
            choices=['trivia_qa', 'squad', 'bioasq', 'nq', 'svamp'],
            help="Dataset to use to assemble few-shot prompt, p_true prompt, and train p_ik.")
        parser.add_argument(
            "--num_samples", type=int, default=400,
            help="Number of samples to use")
        parser.add_argument(
            "--num_few_shot", type=int, default=5,
            help="Number of few shot examples to use")
        parser.add_argument(
            "--p_true_num_fewshot", type=int, default=20,
            help="Number of few shot examples to use")
        parser.add_argument(
            "--p_true_hint", default=False,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--num_generations", type=int, default=1,
            help="Number of generations to use for creating distributions over sentences")
        parser.add_argument(
            "--temperature", type=float, default=0.0,
            help="Temperature (0.0 for greedy decoding)")
        parser.add_argument(
            "--use_mc_options", type=bool, default=True,
            help="Include MC options question?")
        parser.add_argument(
            "--get_training_set_generations", default=True,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--use_context", default=False,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--get_training_set_generations_most_likely_only", default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "Only get embedding of most likely answer for training set. "
                "This is all that's needed for p_true."))
        parser.add_argument('--compute_p_true', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--brief_always", default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--enable_brief", default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--brief_prompt", default='detailed', type=str, 
            choices=['default', 'chat', 'sentence', 'detailed', 'short','manual'],
            help="Type of brief prompt to use for answer generation")
        parser.add_argument(
            "--prompt_type", default='default', type=str)
        parser.add_argument(
            "--compute_uncertainties", default=True,
            action=argparse.BooleanOptionalAction,
            help='Trigger compute_uncertainty_measures.py')
        parser.add_argument(
            "--answerable_only", default=False,
            action=argparse.BooleanOptionalAction,
            help='Exclude unanswerable questions.')

    if 'compute' in stages:
        parser.add_argument('--recompute_accuracy',
                            default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--eval_wandb_runid', type=str,
                            help='wandb run id of the dataset to evaluate on')
        parser.add_argument('--train_wandb_runid', type=str, default=None,
                            help='wandb run id of the dataset from which training embeddings and p_true samples will be taken')
        parser.add_argument('--num_eval_samples', type=int, default=int(1e19))
        parser.add_argument('--compute_predictive_entropy',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_p_ik', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_p_ik_answerable', default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_context_entails_response', default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--analyze_run', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--assign_new_wandb_id', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--restore_entity_eval', type=str, default=entity)
        parser.add_argument('--restore_entity_train', type=str, default=entity)
        parser.add_argument('--condition_on_question',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--strict_entailment',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_all_generations', default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_num_generations', type=int, default=-1)
        parser.add_argument("--entailment_model", default='deberta', type=str)
        parser.add_argument(
            "--entailment_cache_id", default=None, type=str,
            help='Restore entailment predictions from previous run for GPT-4/LLaMa-Entailment.')
        parser.add_argument('--entailment_cache_only', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_p_true_in_compute_stage',
                            default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--reuse_entailment_model',
                            default=False, action=argparse.BooleanOptionalAction,
                            help='Use entailment model as p_true model.')
    return parser


def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # logging.DEBUG


def construct_fewshot_prompt_from_indices(dataset, example_indices, brief, brief_always, make_prompt):
    """Given a dataset and indices, construct a fewshot prompt."""
    if not brief_always:
        prompt = brief
    else:
        prompt = ''

    for example_index in example_indices:

        example = dataset[example_index]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt


def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices


def model_based_metric(predicted_answer, example, model):
    if 'answers' in example:
        correct_answers = example['answers']['text']
    elif 'reference' in example:
        correct_answers = example['reference']['answers']['text']
    else:
        raise ValueError

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"

    if 'gpt' in model.model_name.lower():
        predicted_answer = model.predict(prompt, 0.01)
    else:
        predicted_answer, _, _, _, _ = model.predict(prompt, 0.01)

    if 'yes' in predicted_answer.lower():
        return 1.0
    elif 'no' in predicted_answer.lower():
        return 0.0
    else:
        logging.warning('Redo llm check.')
        predicted_answer, _, _, _, _ = model.predict(prompt, 1)
        if 'yes' in predicted_answer.lower():
            return 1.0
        elif 'no' in predicted_answer.lower():
            return 0.0

        logging.warning('Answer neither no nor yes. Defaulting to no!')
        return 0.0


def llm_metric(predicted_answer, example, model):
    return model_based_metric(predicted_answer, example, model)


def get_gpt_metric(metric_name):

    model_name = '_'.join(metric_name.split('_')[1:])

    class EntailmentGPT():
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, prompt, temperature):
            return oai.predict(prompt, temperature, model=self.model_name)

    gpt_model = EntailmentGPT(model_name)

    def gpt_metric(predicted_answer, example, model):
        del model
        return model_based_metric(predicted_answer, example, gpt_model)

    return gpt_metric


def get_llama_metric(metric_name, max_new_tokens=50):
    """Create a metric using a HuggingFace Llama model as judge.
    
    Args:
        metric_name: Format like 'llm_llama-3-70b' or 'llm_llama-2-13b'
        max_new_tokens: Maximum tokens for judge response
    """
    
    # Extract model name (e.g., 'llm_llama-3.1-70b' -> 'llama-3.1-70b')
    model_name_str = '_'.join(metric_name.split('_')[1:])  # 'llama-3.1-70b' or 'llama-3-70b'
    
    # Convert to proper format expected by HuggingFace:
    # 'Llama-3.1-70B' for Llama-3.1 (becomes meta-llama/Llama-3.1-70B)
    # 'Meta-Llama-3-70B' for Llama-3 (becomes meta-llama/Meta-Llama-3-70B)
    # 'Llama-2-70b-chat-hf' for Llama-2 (becomes meta-llama/Llama-2-70b-chat-hf)
    # Split by '-' and format appropriately
    parts = model_name_str.split('-')  # ['llama', '3.1', '70b'] or ['llama', '3', '70b']
    
    if len(parts) >= 3 and parts[0].lower() == 'llama':
        version_num = parts[1]  # '3.1', '3', or '2'
        size = parts[2]  # '70b', '8b', '13b', etc.
        
        if version_num.startswith('3.1'):
            # Format: 'Llama-3.1-70B' (for meta-llama/Llama-3.1-70B)
            model_name = f'Llama-{version_num}-{size.upper()}'
        elif version_num == '3':
            # Format: 'Meta-Llama-3-70B' (for meta-llama/Meta-Llama-3-70B)
            model_name = f'Meta-Llama-{version_num}-{size.upper()}'
        else:
            # Format: 'Llama-2-70b-chat-hf' (for meta-llama/Llama-2-70b-chat-hf)
            model_name = f'Llama-{version_num}-{size.lower()}-chat-hf'
    elif len(parts) == 2:
        # Format: 'Llama-2-70b-chat-hf' (if someone uses 'llama-2-70b')
        model_name = f'Llama-{parts[1]}-{parts[0].lower()}-chat-hf'
    else:
        model_name = model_name_str.capitalize()
    
    # Auto-add quantization for large models (70B+) to fit in GPU memory
    if '70b' in model_name.lower():
        logging.warning(
            f'70B models require significant GPU memory (~35GB with 8-bit quantization). '
            f'If you have only 8GB GPU, consider using a smaller judge model like Llama-3-8B. '
            f'Attempting to load {model_name} with 8-bit quantization...'
        )
        model_name_with_quant = f'{model_name}-8bit'
    else:
        logging.info(f'Initializing Llama judge model: {model_name}')
        model_name_with_quant = model_name
    
    # Initialize the HuggingFace model for judging
    judge_model = HuggingfaceModel(
        model_name_with_quant, 
        stop_sequences='default',
        max_new_tokens=max_new_tokens
    )
    
    def llama_metric(predicted_answer, example, model):
        del model  # Not used, we use judge_model instead
        return model_based_metric(predicted_answer, example, judge_model)
    
    return llama_metric


def get_reference(example):
    if 'answers' not in example:
        example = example['reference']
    answers = example['answers']
    answer_starts = answers.get('answer_start', [])
    reference = {'answers': {'answer_start': answer_starts, 'text': answers['text']}, 'id': example['id']}
    return reference


def init_model(args):
    mn = args.model_name
    if 'llama' in mn.lower() or 'falcon' in mn or 'mistral' in mn.lower():
        # Use relaxed stop sequences for detailed answers to allow multi-sentence responses
        # For detailed mode, we remove '\n\n' and '\n\n\n' to prevent premature stopping
        if args.brief_prompt == 'detailed':
            stop_sequences = 'detailed'
        else:
            stop_sequences = 'default'
        
        model = HuggingfaceModel(
            mn, stop_sequences=stop_sequences,
            max_new_tokens=args.model_max_new_tokens)
    else:
        raise ValueError(f'Unknown model_name `{mn}`.')
    return model


def get_make_prompt(args):
    if args.prompt_type == 'default':
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if brief_always:
                prompt += brief
            if args.use_context and (context is not None):
                prompt += f"Context: {context}\n"
            prompt += f"Question: {question}\n"
            if answer:
                prompt += f"Answer: {answer}\n\n"
            else:
                prompt += 'Answer:'
            return prompt
    else:
        raise ValueError

    return make_prompt


def get_metric(metric):
    if metric == 'squad':

        squad_metric = load("squad_v2")

        def metric(response, example, *args, **kwargs):
            # Compatibility with recomputation.
            if 'id' in example:
                exid = example['id']
            elif 'id' in example['reference']:
                exid = example['reference']['id']
            else:
                raise ValueError

            prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': exid}
            results = squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)])
            return 1.0 if (results['f1'] >= 50.0) else 0.0

    # Reuses the globally active model for these.
    elif metric == 'llm':
        metric = llm_metric
    elif metric == 'llm_gpt-3.5':
        metric = get_gpt_metric(metric)
    elif metric == 'llm_gpt-4':
        metric = get_gpt_metric(metric)
    elif metric.startswith('llm_llama-'):
        # Use HuggingFace Llama model as judge
        metric = get_llama_metric(metric, max_new_tokens=50)
    else:
        raise ValueError(f'Unknown metric: {metric}')

    return metric


def save(object, file):
    with open(f'{wandb.run.dir}/{file}', 'wb') as f:
        pickle.dump(object, f)
    wandb.save(f'{wandb.run.dir}/{file}')
