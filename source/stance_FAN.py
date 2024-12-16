# Prompt Augmentation Types
# December 11, 2023
# Author: Shivika Sharma

# Importing Libraries

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras import layers
from tqdm import tqdm
import json
import re
from transformers import BatchEncoding
from transformers import BertForMaskedLM, AdamW
from torch.utils.data import DataLoader
import random
import warnings
from sklearn.metrics import precision_recall_fscore_support
warnings.filterwarnings("ignore")


from transformers import BertTokenizer, BertModel
from transformers import TrainingArguments, BertTokenizer, BertForMaskedLM, AdamW, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_metric
from sklearn.metrics import f1_score

from parascore import ParaScorer

from collections import Counter, defaultdict

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms
from biterm.utility import topic_summuary
from biterm.btm import oBTM

from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path

# Importing Dataset
print("\nImporting Datasets")
print(f'------------------------------------------\n')

sem_eval_train = pd.read_csv('./trainingdata-all-annotations.txt', sep="\t", encoding='iso-8859-1', names=['ID', 'Target', 'Tweet', 'Stance', 'Opinion', 'Sentiment'])
sem_eval_train = sem_eval_train.drop(0).reset_index()

# For LexRank
documents = np.array(sem_eval_train['Tweet'].to_list()).flatten()
lxr = LexRank(documents, stopwords=STOPWORDS['en'])

sem_eval_test = pd.read_csv('./testdata-taskA-all-annotations.txt', sep="\t", encoding='iso-8859-1', names=['ID', 'Target', 'Tweet', 'Stance', 'Opinion', 'Sentiment'])
sem_eval_test = sem_eval_test.drop(0).reset_index()

#Targets -  'Atheism' 'Climate Change is a Real Concern' 'Feminist Movement' 'Hillary Clinton' 'Legalization of Abortion'
selected_target = 'Legalization of Abortion' 
sem_eval_train = sem_eval_train[sem_eval_train['Target'] == selected_target]
sem_eval_test = sem_eval_test[sem_eval_test['Target'] == selected_target]
train_len = len(sem_eval_train)

target_topic_dict = {'Atheism':7, 'Climate Change is a Real Concern':5, 'Feminist Movement':7, 'Hillary Clinton':5, 'Legalization of Abortion':3}
target_minority_stance = {'Atheism':'FAVOR', 'Climate Change is a Real Concern':'AGAINST', 'Feminist Movement':'NONE', 'Hillary Clinton':'FAVOR', 'Legalization of Abortion':'FAVOR'}

# Adding BTM Topics
print("\nAdding BTM Topics")
print(f'------------------------------------------\n')

num_topics = target_topic_dict[selected_target]

texts = sem_eval_train['Tweet'].tolist()
vec = CountVectorizer(stop_words='english')
X = vec.fit_transform(texts).toarray()
vocab = np.array(vec.get_feature_names_out())
biterms = vec_to_biterms(X)
btm = oBTM(num_topics=num_topics, V=vocab)

btm_tweet_topics = btm.fit_transform(biterms, iterations=100)
btm_tweet_topics = [btm_tweet_topics[i].argmax() for i in range(len(btm_tweet_topics))]

summary = topic_summuary(btm.phi_wz.T, X, vocab, 10)
btm_topic_top_words = [list(sublist) for sublist in summary['top_words']]

test_biterms = vec_to_biterms(vec.fit_transform(sem_eval_test['Tweet'].tolist()).toarray())
test_topics = btm.transform(test_biterms)
btm_tweet_topics_test = [test_topics[i].argmax() for i in range(len(test_topics))]

sem_eval_train["BTM Topic"] = btm_tweet_topics
sem_eval_test["BTM Topic"] = btm_tweet_topics_test

sem_eval_train = pd.concat([sem_eval_train.sample(frac=1, random_state=36), sem_eval_test], axis=0)
sem_eval_train.reset_index(drop=True, inplace=True)

def target_map(selected_target):
	if selected_target=='Climate Change is a Real Concern':
		return 'CCC'
	if selected_target=='Atheism':
		return 'A'
	if selected_target=='Feminist Movement':
		return 'FM'
	if selected_target=='Hillary Clinton':
		return 'HC'
	if selected_target=='Legalization of Abortion':
		return 'LA' 


# Adding Prompt Template

sem_eval_train['Answered Prompt'] = "The Stance that " + sem_eval_train['Tweet'] + " denotes towards " + sem_eval_train['Target'] + " is " + sem_eval_train['Stance'].str.lower() + "."
sem_eval_train['Prompt'] = "The Stance that " + sem_eval_train['Tweet'] + " denotes towards " + sem_eval_train['Target'] + " is [MASK]."

sem_eval_test['Answered Prompt'] = "The Stance that " + sem_eval_test['Tweet'] + " denotes towards " + sem_eval_test['Target'] + " is " + sem_eval_test['Stance'].str.lower() + "."
sem_eval_test['Prompt'] = "The Stance that " + sem_eval_test['Tweet'] + " denotes towards " + sem_eval_test['Target'] + " is [MASK]."


# Augmentations

mapping_wiki={'Feminist Movement': "The feminist movement, also known as the women's movement, refers to a series of social movements and political campaigns for radical and liberal reforms on women's issues created by the inequality between men and women.[1] Such issues are women's liberation, reproductive rights, domestic violence, maternity leave, equal pay, women's suffrage, sexual harassment, and sexual violence. The movement's priorities have expanded since its beginning in the 1800s, and vary among nations and communities. Priorities range from opposition to female genital mutilation in one country, to opposition to the glass ceiling in another.",
			 'Hillary Clinton': "Hillary Diane Rodham Clinton (born October 26, 1947) is an American politician and diplomat who served as the 67th United States Secretary of State under President Barack Obama from 2009 to 2013, as a U.S. senator representing New York from 2001 to 2009, and as the first lady of the United States as the wife of president Bill Clinton from 1993 to 2001. A member of the Democratic Party, she was the party's nominee in the 2016 U.S. presidential election, becoming the first woman to win a presidential nomination by a major U.S. political party. Clinton won the popular vote, but lost the Electoral College vote, losing the election to Donald Trump.",
			 'Legalization of Abortion': "Abortion laws vary widely among countries and territories, and have changed over time. Such laws range from abortion being freely available on request, to regulation or restrictions of various kinds, to outright prohibition in all circumstances. Many countries and territories that allow abortion have gestational limits for the procedure depending on the reason; with the majority being up to 12 weeks for abortion on request, up to 24 weeks for rape, incest, or socioeconomic reasons, and more for fetal impairment or risk to the woman's health or life. As of 2022, countries that legally allow abortion on request or for socioeconomic reasons comprise about 60% of the world's population.",
			 'Atheism': "Atheism, in the broadest sense, is an absence of belief in the existence of deities. Less broadly, atheism is a rejection of the belief that any deities exist. In an even narrower sense, atheism is specifically the position that there are no deities. Atheism is contrasted with theism, which in its most general form is the belief that at least one deity exists.",
			 'Climate Change is a Real Concern': 'Climate crisis is a term describing global warming and climate change, and their impacts. This term and the term climate emergency have been used to describe the threat of global warming to humanity and the planet, and to urge aggressive climate change mitigation.[2][3][4][5] In the scientific journal BioScience, a January 2020 article, endorsed by over 11,000 scientists worldwide, stated that "the climate crisis has arrived" and that an "immense increase of scale in endeavors to conserve our biosphere is needed to avoid untold suffering due to the climate crisis."'}

# Function that returns augmented prompt 
def template_string(sem_eval_train, index, index1, index2, lbl_add=True, add_wiki=False):
	wiki_txt = ''
	if add_wiki:
		wiki_txt = mapping_wiki[sem_eval_train.loc[index, 'Target']] + " "
	if index1 is None and index2 is None:
		prompt =  wiki_txt + " " +sem_eval_train.loc[index, "Prompt"]
		ans_prompt = wiki_txt + " " +sem_eval_train.loc[index, "Answered Prompt"]
		aug_stat = 'NULL'
	elif lbl_add:
		aug_stat = [sem_eval_train.loc[index1, "Stance"], sem_eval_train.loc[index2, "Stance"]]
		prompt =  wiki_txt + " " + sem_eval_train.loc[index1, "Answered Prompt"]+ " " +sem_eval_train.loc[index2,"Answered Prompt"]+ " "+sem_eval_train.loc[index, "Prompt"]
		ans_prompt =  wiki_txt + " " + sem_eval_train.loc[index1, "Answered Prompt"]+ " " +sem_eval_train.loc[index2,"Answered Prompt"]+ " "+sem_eval_train.loc[index, "Answered Prompt"]
	else:
		aug_stat = [sem_eval_train.loc[index1, "Stance"], sem_eval_train.loc[index2, "Stance"]]
		prompt = wiki_txt + sem_eval_train.loc[index1, "Tweet"]+ " " +sem_eval_train.loc[index2,"Tweet"]+ " "+sem_eval_train.loc[index, "Prompt"]
		ans_prompt = wiki_txt + sem_eval_train.loc[index1, "Tweet"]+ " " +sem_eval_train.loc[index2,"Tweet"]+ " "+sem_eval_train.loc[index, "Answered Prompt"]
	return prompt, ans_prompt, aug_stat

def return_prompt_template(sem_eval_train, index, index1, index2, index3=None, drop=False, lbl_add=True, add_wiki=False):
	if drop==True:
		if index1!=index:
			if index2!=index:
				return template_string(sem_eval_train, index, index1, index2, lbl_add=lbl_add, add_wiki=add_wiki)
			else:
				return template_string(sem_eval_train, index, index1, index3, lbl_add=lbl_add, add_wiki=add_wiki)
		return template_string(sem_eval_train, index, index2, index3, lbl_add=lbl_add, add_wiki=add_wiki)
	else:
		return template_string(sem_eval_train, index, index1, index2, lbl_add=lbl_add, add_wiki=add_wiki)

# Wikipedia Augmentation Function
def add_wiki(sem_eval_train, index, lbl_add=True, add_wiki=True):
	return return_prompt_template(sem_eval_train, index, None, None, lbl_add=lbl_add, add_wiki=add_wiki)

# BTM Tweets
def create_topic_to_tweet_dict(sem_eval_train):
	topic_to_tweet_dict = defaultdict(list)
	for index,row in sem_eval_train.iterrows():
		topic_to_tweet_dict[row['BTM Topic']].append(index)
	return dict(topic_to_tweet_dict)

topic_to_tweet_dict = create_topic_to_tweet_dict(sem_eval_train)

def add_btm_tweet(sem_eval_train, index, lbl_add=True, add_wiki=False):
	topic = sem_eval_train.loc[index, "BTM Topic"]
	topic_to_tweet_dict[topic].remove(index)
	index1 = np.random.choice(topic_to_tweet_dict[topic])
	index2 = np.random.choice(topic_to_tweet_dict[topic])
	topic_to_tweet_dict[topic].append(index)
	return return_prompt_template(sem_eval_train, index, index1, index2, lbl_add=lbl_add, add_wiki=add_wiki)

# BTM Words

def add_btm_words(sem_eval_train, index, lbl_add=False, add_wiki=False):
	topic = sem_eval_train.loc[index, "BTM Topic"]

	exploit_prob = 0.8
	num_exploit = int(exploit_prob*10)
	num_explore = 10 - num_exploit

	exploit_words = random.sample(tuple(btm_topic_top_words[topic]), num_exploit)
	other_words = btm_topic_top_words.copy()
	other_words.pop(topic)
	other_words = np.array(other_words).flatten().tolist()

	explore_words = random.sample(tuple(other_words), num_explore)
	augment_words = exploit_words + explore_words

	augment_words = ' '.join(augment_words)

	wiki_txt = ''
	if add_wiki:
		wiki_txt = mapping_wiki[sem_eval_train.loc[index, 'Target']] + " "

	prompt = wiki_txt + augment_words+ " "+sem_eval_train.loc[index, "Prompt"]
	ans_prompt = wiki_txt+ augment_words+" "+sem_eval_train.loc[index, "Answered Prompt"]

	return prompt, ans_prompt, "NULL"

# SBERT
from sentence_transformers import SentenceTransformer, util

# Load a pre-trained SBERT model
sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

sentences = sem_eval_train['Tweet'].to_list()
sentence_embeddings = sbert.encode(sentences, convert_to_tensor=True)
similarity_matrix = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)

sentences_test = sem_eval_train['Tweet'].to_list()
sentence_embeddings = sbert.encode(sentences, convert_to_tensor=True)
similarity_matrix = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)

def top_sbert(arr):
	indexed_arr = list(enumerate(arr))

	sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)

	highest_indices = [sorted_arr[0][0], sorted_arr[1][0], sorted_arr[2][0]]

	return highest_indices

def add_sbert_template(sem_eval_train, index, lbl_add=True, add_wiki=False):
	index1, index2, index3 = top_sbert(similarity_matrix[index].tolist())
	return return_prompt_template(sem_eval_train, index, index1, index2, index3, drop=True, lbl_add=lbl_add, add_wiki=add_wiki)

# Salience and Diversity

# Returns two tweets from same BTM topic with highest SBERT similarity
def BTM_salience(sem_eval_train, index, req=2, lbl_add=True, add_wiki=False):
	tweet_topic = sem_eval_train.loc[index]['BTM Topic']
	specific_indices = topic_to_tweet_dict[tweet_topic].copy()
	specific_indices.remove(index)
	scores = similarity_matrix[index].tolist()	
	mask = [1.0 if index in specific_indices else 0.0 for index in range(len(scores))]
	masked_scores = [scores[j] * mask[j] for j in range(len(scores))]
	
	top_indices = np.argsort(masked_scores)[-req:]
	index1, index2 = list(top_indices)
	return return_prompt_template(sem_eval_train, index, index1, index2, lbl_add=lbl_add, add_wiki=add_wiki)

# Returns #'req' tweets from same BTM topic with highest SBERT similarity
def salience(sem_eval_train, tweet_topic, index=None, req=2, similarity_matrix=similarity_matrix):
	indices = topic_to_tweet_dict[tweet_topic].copy()
	if index is not None:
		indices.remove(index)
	return find_highest_similar_indices(similarity_matrix, index, indices, bottom_n=req,random_sampling=False)

def find_highest_similar_indices(matrix, i, specific_indices, bottom_n=1, random_sampling=False):
	scores = matrix[i].tolist()
	mask = [1.0 if index in specific_indices else 0.0 for index in range(len(scores))]
	masked_scores = [scores[j] * mask[j] for j in range(len(scores))]

	if random_sampling:
		bottom_indices = np.argsort(masked_scores)[-1*min(5,len(specific_indices)):]
		bottom_indices = np.random.choice(bottom_indices)
		return [int(bottom_indices)]

	bottom_indices = np.argsort(masked_scores)[-bottom_n:]
	return list(bottom_indices)
	
# Finds #'req' most diverse tweets in the corpus
def return_diversity_indices(sem_eval_train, req=10, lxr=lxr):
	sentences = sem_eval_train['Tweet'].to_list()
	scores_cont = lxr.rank_sentences(sentences,threshold=None,fast_power_method=False)
	sorted_indices = np.argsort(scores_cont)
	top_indices = sorted_indices[-req:][::-1]
	return top_indices.tolist()

diversity_indices = return_diversity_indices(sem_eval_train, req=10)

# 1 tweet from salience and 1 from diversity
def salience_and_diversity_old(sem_eval_train, index, req=1, lbl_add=True, add_wiki=False):
	index1 = salience(sem_eval_train, sem_eval_train.loc[index]['BTM Topic'], index, req=1)[0]
	index3 = np.random.choice(diversity_indices) #diversity(sem_eval_train)[0]
	index3 = diversity_indices.index(index3)
	while diversity_indices[index3] == index or diversity_indices[index3] == index1:
		index3 = (index3 + 1)%len(diversity_indices)
	index3 = diversity_indices[index3]
	#index1.extend(index2)
	return return_prompt_template(sem_eval_train, index, index1, index3, lbl_add=lbl_add, add_wiki=add_wiki)

# Proposed approach - adding two tweets from salience for minority class, otherwise follows salience_and_diversity_old
def salience_and_diversity(sem_eval_train, index, req=1, lbl_add=True, add_wiki=False):
	if sem_eval_train.loc[index, "Stance"] != target_minority_stance[selected_target]:
		return salience_and_diversity_old(sem_eval_train, index, lbl_add=lbl_add, add_wiki=add_wiki)
	index1, index2 = salience(sem_eval_train, sem_eval_train.loc[index]['BTM Topic'], index, req=2)
	index3 = np.random.choice(diversity_indices) #diversity(sem_eval_train)[0]
	index3 = diversity_indices.index(index3)
	while diversity_indices[index3] == index or diversity_indices[index3] == index1 or diversity_indices[index3] == index2:
		index3 = (index3 + 1)%len(diversity_indices)
	index3 = diversity_indices[index3]
	#index1.extend(index2)
	return return_prompt_template_new(sem_eval_train, index, index1, index2, index3, lbl_add=lbl_add, add_wiki=add_wiki)

# Returns augmented prompt when given three tweets to augment
def return_prompt_template_new(sem_eval_train, index, index1, index2, index3, lbl_add=True, add_wiki=False):
	wiki_txt = ''
	if add_wiki:
		wiki_txt = mapping_wiki[sem_eval_train.loc[index, 'Target']] + " "
	if index1 is None and index2 is None:
		prompt =  wiki_txt + " " +sem_eval_train.loc[index, "Prompt"]
		ans_prompt = wiki_txt + " " +sem_eval_train.loc[index, "Answered Prompt"]
		aug_stat = 'NULL'
	elif lbl_add:
		aug_stat = [sem_eval_train.loc[index1, "Stance"], sem_eval_train.loc[index2, "Stance"], sem_eval_train.loc[index3, "Stance"]]
		prompt =  wiki_txt + " " + sem_eval_train.loc[index1, "Answered Prompt"]+ " " +sem_eval_train.loc[index2,"Answered Prompt"]+ " "+ sem_eval_train.loc[index3, "Answered Prompt"]+ " " +sem_eval_train.loc[index, "Prompt"]
		ans_prompt =  wiki_txt + " " + sem_eval_train.loc[index1, "Answered Prompt"]+ " " +sem_eval_train.loc[index2,"Answered Prompt"]+ " "+ sem_eval_train.loc[index3, "Answered Prompt"]+ " " +sem_eval_train.loc[index, "Answered Prompt"]
	else:
		aug_stat = [sem_eval_train.loc[index1, "Stance"], sem_eval_train.loc[index2, "Stance"]]
		prompt = wiki_txt + sem_eval_train.loc[index1, "Tweet"]+ " " +sem_eval_train.loc[index2,"Tweet"]+ " "+sem_eval_train.loc[index3, "Tweet"]+ " " +sem_eval_train.loc[index, "Prompt"]
		ans_prompt = wiki_txt + sem_eval_train.loc[index1, "Tweet"]+ " " +sem_eval_train.loc[index2,"Tweet"]+ " "+sem_eval_train.loc[index3, "Tweet"]+ " " +sem_eval_train.loc[index, "Answered Prompt"]
	return prompt, ans_prompt, aug_stat

# Augmenting Data
def data_augmentation(sem_eval_train_, exploit_prob, exploit_func, lbl_add=True, add_wiki=False):
	sem_eval_train = sem_eval_train_.copy()
	sem_eval_train['Prompt template filled'] = None
	sem_eval_train['Prompt template'] = None
	sem_eval_train['Augment Stance'] = None
	answered_prompts = sem_eval_train['Answered Prompt'].to_list()
	for index, row in tqdm(sem_eval_train.iterrows()):
		if np.random.uniform() < exploit_prob:
			prompt, ans_prompt, aug_stat = exploit_func(sem_eval_train, index, lbl_add=lbl_add, add_wiki=add_wiki)
			sem_eval_train.at[index, 'Prompt template filled'] = ans_prompt
			sem_eval_train.at[index, 'Prompt template'] = prompt
			sem_eval_train.at[index, 'Augment Stance'] = aug_stat
		else:
			index1 = np.random.randint(len(sem_eval_train)-1)
			index2 = np.random.randint(len(sem_eval_train)-1)
			prompt, ans_prompt, aug_stat = return_prompt_template(sem_eval_train, index, index1, index2, lbl_add=lbl_add, add_wiki=add_wiki)
			sem_eval_train.at[index, 'Prompt template filled'] = ans_prompt
			sem_eval_train.at[index, 'Prompt template'] = prompt
			sem_eval_train.at[index, 'Augment Stance'] = aug_stat
	return sem_eval_train

"""
For random - data_augmentation(sem_eval_train, 0, None, lbl_add=True, add_wiki=False) 
For SBERT only - data_augmentation(sem_eval_train, 1, add_sbert_template, lbl_add=True, add_wiki=False)  
For SBERT with random - data_augmentation(sem_eval_train, 0.4, add_sbert_template, lbl_add=True, add_wiki=False)  
For BTM Tweet - data_augmentation(sem_eval_train, 1, add_btm_tweet, lbl_add=True, add_wiki=False)  
For BTM words -  data_augmentation(sem_eval_train, 1, add_btm_words, lbl_add=True, add_wiki=False)  
For Salience and Diversity with Minority - data_augmentation(sem_eval_train, 1, salience_and_diversity_old, lbl_add=True, add_wiki=False)  
For Salience and Diversity - data_augmentation(sem_eval_train, 1, salience_and_diversity), lbl_add=True, add_wiki=False) 
For Wikipedia -  data_augmentation(sem_eval_train, 1, add_wiki, lbl_add=True, add_wiki=True)
For BTM Salience - data_augmentation(sem_eval_train, 1, BTM_salience, lbl_add=True, add_wiki=False) 
"""
print("\nAugmenting Data")
print(f'------------------------------------------\n')
augment_type = 'three_SaLD_5'
sem_eval_train_augmented = data_augmentation(sem_eval_train,1, salience_and_diversity)       

print("\nTrain - Test Split")
print(f'------------------------------------------\n')
#train_len = int(0.8*len(sem_eval_train_augmented))
sem_eval_test_augmented = sem_eval_train_augmented[train_len:].reset_index()
sem_eval_train_augmented = sem_eval_train_augmented[:train_len]

# Initialising Model

class My_BERT(nn.Module):
	def __init__(self, model_name, tokenizer, answer_tokens, c_tokens=None, is_map=False, aggregate=False):
		super(My_BERT, self).__init__()
		self.BERT = BertForMaskedLM.from_pretrained(model_name)
		self.tokenizer = tokenizer
		for param in self.BERT.parameters():
			param.requires_grad = True

		self.answer_ids = self.tokenizer.encode(answer_tokens, add_special_tokens=False)
		self.N = len(answer_tokens)
		self.mask_token_id = 103
		self.loss_func = nn.CrossEntropyLoss()
		self.is_map = False
		if is_map:
			self.class_tokens = [self.tokenizer.encode(c_tk, add_special_tokens=False) for c_tk in c_tokens]
			self.is_map = True
		if aggregate:
			self.num_aggregate = 3

	# Returns {loss, class probabilities}
	def forward(self, input_id, attention_mask, input_label):
			outputs = self.BERT(input_ids=input_id,attention_mask=attention_mask)
			out_logits = outputs.logits

			mask_position = input_id.eq(self.mask_token_id)
			mask_logits = out_logits[mask_position, :].view(out_logits.size(0), -1, out_logits.size(-1))[:, -1, :]

			answer_logits = mask_logits[:, self.answer_ids]
			if self.is_map:
				for c_index in range(len(self.class_tokens)):
					answer_logits[0][c_index] = torch.sum(mask_logits[:, self.class_tokens[c_index]])

			answer_probs = answer_logits.softmax(dim=1)
			loss = self.loss_func(answer_logits, input_label)

			return {'loss':loss, 'answer_probs':answer_probs}

	# Returns [class probabilities, predicted token]
	def predict(self, input_id, attention_mask):
		answer_probs = []
		for _ in range(self.num_aggregate):
			answer_probs.append(self.forward(input_id, attention_mask, torch.tensor([random.randint(0,self.N-1)]).to(device))['answer_probs'])
		predictions = [self.tokenizer.convert_ids_to_tokens(max(list(zip(self.answer_ids, inst[0])), key=lambda x: x[1])[0]) for inst in answer_probs]
		answer_probs = torch.mean(torch.stack(answer_probs), dim=0)
		token_probs = list(zip(self.answer_ids, answer_probs[0]))
		pred_token = self.tokenizer.convert_ids_to_tokens(max(token_probs, key=lambda x: x[1])[0])
		token_probs = [(self.tokenizer.convert_ids_to_tokens(ans_tk),ans_prob.item()) for ans_tk,ans_prob in token_probs]

		return [sorted(token_probs, key=lambda x: x[1], reverse=True), pred_token, predictions]


def custom_model(model_name, answer_tokens, c_tokens=None, aggregate=False):
	tokenizer = BertTokenizer.from_pretrained(model_name)

	if c_tokens is not None:
		is_map = True
	else:
		is_map = False
	model = My_BERT(model_name, tokenizer, answer_tokens, c_tokens=c_tokens, is_map=is_map, aggregate=aggregate)

	return model, tokenizer

print("\nInitialising Model")
print(f'------------------------------------------\n')
c_tokens = [['favor','favouring','positive','good','supporting'],['against','opposing','resist','contradict','bad','disfavor'],['neutral','none','unbiased','impartial']]
model, tokenizer = custom_model('bert-base-uncased', ['favor','against','none'], c_tokens, aggregate=True)


sem_eval_train_augmented['Label'] = None
class_labels = ['favor','against','none']
for index,row in sem_eval_train_augmented.iterrows():
	sem_eval_train_augmented.at[index,'Label'] = class_labels.index(row['Stance'].lower())

class Dataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels
	def __getitem__(self, idx):
		item_dict =  {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item_dict['labels'] =  self.labels[idx]
		return item_dict
	def __len__(self):
		return len(self.encodings.input_ids)

inputs = tokenizer(sem_eval_train_augmented['Prompt template'].to_list(),return_tensors='pt',padding='max_length',truncation=True,max_length=512 )
dataset = Dataset(inputs, sem_eval_train_augmented['Label'].to_list())
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

print("\nTraining Model")
print(f'------------------------------------------\n')

num_epochs = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
optim = AdamW(model.parameters(), lr=2e-6)
model.to(device)
model.train()

train_losses = []
train_losses_file_name = "./"+target_map(selected_target)+"_training_loss_epochs_"+str(num_epochs)+"_"+augment_type+".txt"

# Training
for epoch in range(num_epochs):
	loop = tqdm(loader, leave=True)
	epoch_losses = []
	for batch in loop:
		optim.zero_grad()
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['labels'].to(device)
		outputs = model(input_ids, attention_mask,labels)
		loss = outputs['loss']
		epoch_losses.append(loss.item())
		loss.backward()
		optim.step()
		loop.set_description(f'Epoch {epoch}')
		loop.set_postfix(loss=loss.item())
	train_losses.append(np.mean(epoch_losses))

with open(train_losses_file_name, 'w') as file:
	json.dump(train_losses, file)

# Getting predictions
sem_eval_train_augmented = sem_eval_train_augmented.reset_index()

print("\nPredicting Stance")
print(f'------------------------------------------\n')

def predict_results(sem_eval_train_augmented):
	sem_eval_train_augmented['Predicted Token'] = None
	sem_eval_train_augmented['Class Probabilities'] = None
	sem_eval_train_augmented['All Predictions'] = None
	for index,row in tqdm(sem_eval_train_augmented.iterrows()):
		tokenized_tweet = tokenizer(row['Prompt template'], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
		probs, pred_token, predictions = model.predict(tokenized_tweet['input_ids'].to(device), tokenized_tweet['attention_mask'].to(device))
		sem_eval_train_augmented.at[index, 'Predicted Token'] = pred_token 
		sem_eval_train_augmented.at[index, 'Class Probabilities'] =  probs 
		sem_eval_train_augmented.at[index, 'All Predictions'] = predictions
	return sem_eval_train_augmented

print("Train Data: ",sep='')
sem_eval_train_augmented = predict_results(sem_eval_train_augmented)
print("Test Data: ",sep='')
sem_eval_test_augmented = predict_results(sem_eval_test_augmented)



def print_result(sem_eval_train_augmented):
	y_true = [x[0].lower() for x in sem_eval_train_augmented['Stance'].to_list()]
	y_pred = sem_eval_train_augmented['Predicted Token'].to_list()
	metric_dict = {}
	
	tmp = list(zip(sem_eval_train_augmented['Stance'].to_list(), sem_eval_train_augmented['Predicted Token'].to_list()))
	y_true = [x[0].lower() for x in tmp]
	y_pred = [x[1] for x in tmp]
	print(f"F1 score: {f1_score(y_true, y_pred, average='weighted')}")
	metric_dict['F1'] = f1_score(y_true, y_pred, average='weighted')

	for class_lbl in set(y_true):
		class_dict = {}
		precision, recall, F1 = precision_recall_fscore_support(y_true, y_pred, labels=[class_lbl], average='weighted')[:-1]
		class_dict['Precision'] = precision
		class_dict['Recall'] = recall
		class_dict['F1'] = F1
		metric_dict[class_lbl] = class_dict
		print(f"\tClass Label - {class_lbl}\n\t\tPrecision - {precision} \n\t\tRecall - {recall} \n\t\tF1 - {F1}")

	return metric_dict


print("\nFinal Summary")
print(f'------------------------------------------\n')
print("Training Data: ")
metric_dict_train = print_result(sem_eval_train_augmented)
train_metric_file_name = "./"+target_map(selected_target)+"_"+"epochs_"+str(num_epochs)+"_"+augment_type+"_train.txt"

with open(train_metric_file_name, 'w') as file:
	json.dump(metric_dict_train, file)


print("Test Data: ")
metric_dict_test = print_result(sem_eval_test_augmented)

test_metric_file_name = "./"+target_map(selected_target)+"_"+"epochs_"+str(num_epochs)+"_"+augment_type+"_test.txt"

with open(test_metric_file_name, 'w') as file:
	json.dump(metric_dict_test, file)

test_df_file_name = "./"+target_map(selected_target)+"_"+"epochs_"+str(num_epochs)+"_"+augment_type+"_test_df.csv"
sem_eval_test_augmented.to_csv(test_df_file_name)
train_df_file_name = "./"+target_map(selected_target)+"_"+"epochs_"+str(num_epochs)+"_"+augment_type+"_train_df.csv"
sem_eval_train_augmented.to_csv(train_df_file_name)

print("\nEnd")
print(f'------------------------------------------\n')

