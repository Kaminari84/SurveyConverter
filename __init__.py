import logging
import os
import json
import requests
import random
import base64

import time
import pytz
from datetime import datetime, timedelta
from pytz import timezone
from pytz import common_timezones
from pytz import country_timezones

from flask import Flask, request, make_response, render_template, current_app
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy

from probReactionChoices import probReactionChoices
from reactions import reaction_repo

from m_classicTextClf import TextClassifier
import spacy

from flask_cors import CORS

logging.basicConfig( filename='/var/www/testapp/logs/app_'+time.strftime('%d-%m-%Y-%H-%M-%S')+'.log', level=logging.INFO)

logging.info("Server loading...")

app = Flask(__name__)
app.app_context().push()
UPLOAD_FOLDER = '/static/audio'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:root@127.0.0.1:3306/test"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_POOL_RECYCLE'] = 1

db = SQLAlchemy(app)

#DATABASE Log class
class EventLog(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	conv_id = db.Column(db.String(80))
	event = db.Column(db.String(1024))
	timestamp = db.Column(db.DateTime())

	def __init__(self, conv_id, event):
		self.conv_id = conv_id
		self.event = event
		self.timestamp = pstnow()

#Date-Time helpers
def utcnow():
    return datetime.now(tz=pytz.utc)

def pstnow():
    utc_time = utcnow()
    pacific = timezone('US/Pacific')
    pst_time = utc_time.astimezone(pacific)
    return pst_time

#Debugging POST reqyest for TTS helper
def pretty_print_POST(req):
	"""
	At this point it is completely built and ready
	to be fired; it is "prepared".

	However pay attention at the formatting used in 
	this function because it is programmed to be pretty 
	printed and may differ from the actual request.
	"""
	print('{}\n{}\n{}\n\n{}'.format(
		'-----------START-----------',
		req.method + ' ' + req.url,
		'\n'.join('{}: {}'.format(k, v) for k, v in req.headers.items()),
		req.body,
	))

def initReactionRepo():
	for key, value in reaction_repo.items():
		if "prefix" in value:
			mgr = probReactionChoices(value['prefix'])
			current_app.reactionMgrs[key] = mgr
		else:
			mgr = probReactionChoices(value)
			current_app.reactionMgrs[key] = mgr

#Server initial setup
def setup_app(app):

	#logging.info("Test loading model training data...")
	#f = open(os.path.join(app.root_path,'static/model_data/q_framing_data.json'), 'r')
	#for line in f:
	#	logging.info(line)
	#f.close()


	audio_path = os.path.join(app.root_path, "static/audio/audio_test.wav")
	with open(audio_path, 'wb') as fd:
		pass

	logging.info("Instance path:"+os.path.join(app.root_path))
	logging.info('Creating all database tables...')
	db.create_all()

	with app.app_context():
		# within this block, current_app points to app.
		logging.info("App name:"+str(current_app.name))
		current_app.reactionMgrs = {}

		initReactionRepo()

		#POS tagger
		current_app.nlp = spacy.load("en_core_web_sm")
		current_app.nlp.vocab["indicate"].is_stop = True
		current_app.nlp.vocab["extent"].is_stop = True

		#train question framing classifier
		logging.info("---------QUESTION FRAMING CLASSIFIER----------")
		data_path = os.path.join(app.root_path, "static/model_data/q_framing_data.json")
		model_path = os.path.join(app.root_path, "static/models/q_framing_model")
		current_app.q_clf = TextClassifier(data_path, model_path)
		#current_app.q_clf.train(test_prop=0.1)

		#train answer framing classifier
		logging.info("---------ANSWER SENTIMENT CLASSIFIER----------")
		data_path = os.path.join(app.root_path, "static/model_data/ans_types.json")
		model_path = os.path.join(app.root_path, "static/models/ans_types_model")
		current_app.ans_clf = TextClassifier(data_path, model_path)
		#current_app.ans_clf.train(test_prop=0.1)

		#train survey domain recongition classifier
		logging.info("---------SURVEY DOMAIN CLASSIFIER----------")
		data_path = os.path.join(app.root_path, "static/model_data/domain_survey_data.json")
		model_path = os.path.join(app.root_path, "static/models/domain_survey_model")
		current_app.d_clf = TextClassifier(data_path, model_path, rem_stop_words=True)
		#current_app.d_clf.train(test_prop=0.1, rem_stop_words=True)

		#train survey domain recongition classifier
		logging.info("---------QUESTION PREFIX CLASSIFIER----------")
		data_path = os.path.join(app.root_path, "static/model_data/q_prefix_data.json")
		model_path = os.path.join(app.root_path, "static/models/q_prefix_model")
		current_app.prfx_clf = TextClassifier(data_path, model_path)
		#current_app.prfx_clf.train(test_prop=0.1)

	logging.info('Done!')
	logging.info("Start the actual server...")


setup_app(app)

@app.route('/hello')
def hello_world():
	source = request.args.get('source')

	logging.info("Hello server route called, source"+str(source))
	return 'Hello, Survey Converter from:'+str(source)

@app.route('/')
def conv_survey():
	survey_files = os.listdir(os.path.join(app.root_path,'static/surveys/'))

	return render_template('bot_interface.html', surveys = survey_files )

#Convert given survey
@app.route("/augment_survey", methods = ['POST'])
def augment_survey():
	logging.info("Trying to agument the survey...")

	survey_file = request.form.get('survey_file')
	logging.info("Survey file:"+str(survey_file))

	survey_questions = ""
	with open(os.path.join(app.root_path, 'static/surveys',survey_file), 'r', encoding='utf-8') as f:
		survey_questions = json.load(f)

	# Augmentation params
	isOpening = True if request.form.get('isOpening') == "true" else False # Is intro added?
	isClosing = True if request.form.get('isClosing') == "true" else False # Is closing added?

	logging.info("IS OPENING:"+str(isOpening))
	logging.info("IS CLOSING:"+str(isClosing))

	progress_repeat = int(request.form.get('progressRepeatN')) # Every how many questions does the progress reporting repeat
	reaction_repeat = int(request.form.get('reactionRepeatN')) # Every how many questions does the reaction happen

	logging.info("REACTION REPEAT EVERY:"+str(reaction_repeat))
	logging.info("PROGRESS REPEAT EVERY:"+str(progress_repeat))

	# Empathy: 1.0 - all reactins are empathy loaded, 0.0 - all reactions are neutral
	reaction_empathy = float(request.form.get('empathyLevel'))
	logging.info("EMPATHY LEVEL:"+str( reaction_empathy))

	# Q Augmentation: 1.0 - all question augmented, 0.0 - none augmented
	q_augment = float(request.form.get('qAugmentLevel'))
	logging.info("QUESTION AUGMENTATION LEVEL:"+str(q_augment))

	logging.info("-------ORIGINAL--------")
	# Show the original survey
	i = 0
	all_survey_words = ""
	for question in survey_questions:
		i += 1
		logging.info(str(i)+"."+str(question['text'])+ ", A: "+str(question['type']))
		#rem stop words
		sent_tokens = current_app.nlp(question['text'].lower())
		for token in sent_tokens:
			if token.is_stop:
				pass
			else:
				all_survey_words += token.text+" "

	logging.info("DOMAIN ["+str(len(all_survey_words))+"]->"+str(all_survey_words))
	# Recognize the survey domain
	topic = "unknown"
	with app.app_context():
		topic = str(current_app.d_clf.classify([all_survey_words])[0])

	logging.info("Detected domain:"+str(topic))

	# Deep content params
	name = str(topic).title().replace(" ","")+"Bot"
	logging.info("Bot name:"+str(name))
	#sections = ["Habitual Action", "Understanding", "Reflection", "Critical Reflection"]

	#Add survey progress blocks
	logging.info("-------PROCESSING-------")
	s_len = len(survey_questions)
	i = 0
	qn = 0
	with app.app_context():
		itr_questions = iter(survey_questions)
		for question in itr_questions:
			if "//" in question:
				logging.info("Comment found")

			### Reactions ##
			if reaction_repeat > 0 and i % reaction_repeat == (reaction_repeat-1):
				react_empathicly = (random.random() <= reaction_empathy)
				#print("Rand:", react_empathicly)
				#print("Q:",question)

				reaction_block = {}

				#Add empathetic reaction that matches the context (question-answer)
				if react_empathicly:
					q_frame = callQuestionFraming(question['text'])
					logging.info("Frame:"+str(q_frame))
					question['framing'] = q_frame

					# For Yes/No type
					if question['type'] == 'Yes/No':
						positive_react = current_app.reactionMgrs['reaction_positive_list'].getLeastFreqChoice() #random.choice(reaction_positive_list)
						negative_react = current_app.reactionMgrs['reaction_negative_list'].getLeastFreqChoice() #random.choice(reaction_negative_list)

						if q_frame == "Positive_frame":
							reaction_block['Yes'] = positive_react
							reaction_block['No'] = negative_react
						elif q_frame == "Negative_frame":
							reaction_block['Yes'] = negative_react
							reaction_block['No'] = positive_react

					 # For Options
					if question['type'] == 'Options':
						react = current_app.reactionMgrs['reaction_neutral_list'].getLeastFreqChoice() #random.choice(reaction_neutral_list)
						logging.info("Orig neutr react:"+str(react))
						for option in question['options']:
							logging.info("Option", option['text'])

							ans_frame = current_app.ans_clf.classify([option['text']])[0]
							option['framing'] = ans_frame

							# Positive Answers
							#if option['text'] in pos_answers:
							if ans_frame == "pos_answer":
								if q_frame == "Positive_frame":
									react = current_app.reactionMgrs['reaction_positive_list'].getLeastFreqChoice() #random.choice(reaction_positive_list)
								elif q_frame == "Negative_frame":
									react = current_app.reactionMgrs['reaction_negative_list'].getLeastFreqChoice() #random.choice(reaction_negative_list)
							# Negative Answers
							#elif option['text'] in neg_answers:
							if ans_frame == "neg_answer":
								if q_frame == "Positive_frame":
									react = current_app.reactionMgrs['reaction_negative_list'].getLeastFreqChoice() #random.choice(reaction_negative_list)
								elif q_frame == "Negative_frame":
									react = current_app.reactionMgrs['reaction_positive_list'].getLeastFreqChoice() #random.choice(reaction_positive_list)
							# Neutral Answers or Unknown
							else:
								pass

							#Assign reaction to option
							reaction_block[option['value']] = react
				else: #if react_empatically
					# For Yes/No type
					if question['type'] == 'Yes/No':
						neutral_react =  current_app.reactionMgrs['reaction_neutral_list'].getLeastFreqChoice() #random.choice(reaction_neutral_list)
						reaction_block['Yes'] = neutral_react
						reaction_block['No'] = neutral_react
					# For Options
					if question['type'] == 'Options':
						react = current_app.reactionMgrs['reaction_neutral_list'].getLeastFreqChoice() #random.choice(reaction_neutral_list)
						for option in question['options']:
							reaction_block[option['value']] = react

				if bool(reaction_block):
					question['reactions'] = reaction_block

			#### Question augmentation ###
			if q_augment > 0 and question['type'] != "Skip" and "mod_text" not in question:
				addPrefix = (random.random() <= q_augment)
				if addPrefix:
					prefix_cat = current_app.prfx_clf.classify([question['text']])[0]
					question['prefix_class'] = prefix_cat

					if prefix_cat != "q_none":
						#'q_verb_prefix'
						prefix = current_app.reactionMgrs[prefix_cat].getLeastFreqChoice() #random.choice(reaction_positive_list)
						conv_rules = reaction_repo[prefix_cat]['conv']

						if i>0:
							addRepeat = (random.random() <= 0.7)
							if addRepeat:
								repeat = current_app.reactionMgrs['repeat_prefix_additions'].getLeastFreqChoice()
								prefix = repeat+", "+prefix[0].lower()+prefix[1:]

						#Conversions to the question text
						text = question['text'][0].lower()+question['text'][1:]
						for conv in conv_rules:
							if conv['type'] == "replace":
								text = text.replace(conv['from'], conv['to'])

						question['org_text'] = question['text']
						question['prefix'] = prefix
						question['mod_text'] = text
						question['text'] = prefix+" "+text

			#### Progress ####
			if progress_repeat > 0 and i % progress_repeat == (progress_repeat-1):
				pos = round((qn*100)//s_len)
				progress_text = current_app.reactionMgrs['progress_list'].getLeastFreqChoice()  #random.choice(progress_list)
				if (pos > 40 and pos < 60):
					progress_text = current_app.reactionMgrs['progress_middle_list'].getLeastFreqChoice() #random.choice(progress_middle_list)
				elif (pos > 85):
					progress_text = current_app.reactionMgrs['progress_end_list'].getLeastFreqChoice() #random.choice(progress_end_list)

				block = {}
				block['text'] = progress_text.replace("{d}", str(qn)).replace("{n}",str(s_len)).replace("{l}", str(s_len-qn)).replace("{percent}",str(round(qn*10//s_len*10)))
				block['type'] = "Skip"
				block['source'] = "augmented"

				survey_questions.insert(i, block)

				next(itr_questions, None)

				i+=1

			i += 1
			qn += 1

		#### Add beginning ####
		if (isOpening):
			logging.info("Adding OPENING block")
			block = {}
			close_text = current_app.reactionMgrs['intro_list'].getLeastFreqChoice() #random.choice(intro_list)
			block['text'] = close_text.replace("{name}",name).replace("{topic}",topic)
			block['type'] = "Skip"
			block['source'] = "augmented"

			survey_questions.insert(0, block)

		#### Add ending ####
		if (isClosing):
			logging.info("Adding CLOSING block")
			block = {}
			close_text = current_app.reactionMgrs['close_list'].getLeastFreqChoice() #random.choice(close_list)
			block['text'] = close_text
			block['type'] = "End"
			block['source'] = "augmented"

			survey_questions.append(block)

		#-----------OTHER AUGMENTATIONS GO HERE--------------------


	#Save to json file
	conv_survey_file = survey_file.replace('.json','_conv.json')
	logging.info("Saving conversational survey file:"+str(conv_survey_file))

	with open(os.path.join(app.root_path, 'static/conv_surveys', conv_survey_file), 'w', encoding='utf-8') as f:
		str_json = json.dumps(survey_questions, indent=4, sort_keys=True)
		#str_json = str_json.replace("'","\\'")
		#str_json = str_json.replace("\'","\\'")
		print(str_json, file=f)

	#Save in HarborBot language format
	clean_survey_questions = []
	for old_q in survey_questions:
		new_q = {}
		new_q['text'] = { 'en': old_q['text'] }
		new_q['type'] = old_q['type']

		if old_q['type'] == 'Options' or old_q['type'] == 'Options-multiple':
			new_q['options'] = { 'en': [] }
			for option in old_q['options']:
				new_q['options']['en'].append({
					'text': option['text'],
					'value': option['value']
				})

		if 'reactions' in old_q:
			new_q['reactions'] = {}
			for reaction in old_q['reactions']:
				new_q['reactions'][reaction] = { 'en': old_q['reactions'][reaction] }

		clean_survey_questions.append(new_q)

	with open(os.path.join(app.root_path, 'static/clean_conv_surveys', conv_survey_file), 'w', encoding='utf-8') as f:
		str_json = json.dumps(clean_survey_questions, indent=4, sort_keys=True)
		print(str_json, file=f)


	cov_survey = survey_questions
	json_resp = json.dumps({'status': 'OK', 'message':'',
				'survey_data':cov_survey,
				'conv_survey_file': conv_survey_file,
				'domain':topic, 'bot_name':name})

	return make_response(json_resp, 200, {"content_type":"application/json"})


#Get list of surveys in storage
@app.route('/get_survey_list')
def get_survey_list():
	print('Get survey list called:')

	survey_files = os.listdir(os.path.join(app.root_path, 'static/surveys/'))
	json_resp = json.dumps({'status': 'OK', 'message':'', 'surveys':survey_files})

	return make_response(json_resp, 200, {"content_type":"application/json"})

#Question framing REST
@app.route("/q_framing", methods = ['GET'])
def q_framing():
	logging.info("Getting framing for question...")

	q = request.args.get('q')
	logging.info("q="+str(q))

	q_frame = callQuestionFraming(q)
	logging.info("Frame:"+str(q_frame))

	json_resp = json.dumps({'status': 'OK',
				'message':'',
				'q': q,
				'q_framing':q_frame})

	return make_response(json_resp, 200, {"content_type":"application/json"})


#Call different services to classify the questions framing
def callQuestionFraming(q, model="classic"):
	if model == "classic":
		labels = current_app.q_clf.classify([q])
		return labels[0]
	elif model == "rasa":
		return None
	elif model == "py_porch":
		return None
	elif model == "luis":
		return None
	else:
		return None

#Answer framing RESRT endpoint
@app.route("/ans_framing", methods = ['GET'])
def ans_framing():
	logging.info("Getting framing for answer options...")

	ans = request.args.get('a')
	logging.info("a="+str(ans))

	ans_frame = current_app.ans_clf.classify([ans])[0]
	logging.info("Frame:"+str(ans_frame))

	json_resp = json.dumps({'status': 'OK', 
				'message':'', 
				'a': ans,
				'ans_framing':ans_frame})

	return make_response(json_resp, 200, {"content_type":"application/json"})

#get prefix classification REST endpoint
@app.route("/prefix_class", methods = ['GET'])
def prefix_class():
	logging.info("Getting class for the appropriate prefix...")

	q = request.args.get('q')
	logging.info("q="+str(q))

	prefix_cat = current_app.prfx_clf.classify([q])[0]
	logging.info("Prefix cat:"+str(prefix_cat))

	#get examples of matches
	prefixes = []
	conv_rules = []
	aug_text = []
	if prefix_cat != "q_none":
		prefixes = reaction_repo[prefix_cat]['prefix']
		conv_rules = reaction_repo[prefix_cat]['conv'] 

		#Conversions to the augmented text
		text = q[0].lower()+q[1:]
		for conv in conv_rules:
			if conv['type'] == "replace":
				text = text.replace(conv['from'], conv['to'])
		aug_text = random.choice(prefixes)+" "+text

	json_resp = json.dumps({'status': 'OK',
				'message':'',
				'q': q,
				'prefix_class': prefix_cat,
				'matching_prefixes': prefixes,
				'conversion_rules': conv_rules,
				'conv_question': aug_text})

	return make_response(json_resp, 200, {"content_type":"application/json"})

#get selected survey content
@app.route("/get_survey")
def get_survey():
	logging.info("Trying to get survey..")

	survey_file = request.args.get('survey_file')
	survey_source = request.args.get('survey_source')

	json_resp = json.dumps({'status': 'ERROR', 'message':''})

	if survey_file != None and survey_source != None:
		logging.info("Survey file:"+str(survey_file))
		logging.info("Survey source:"+str(survey_source))

		survey_path = os.path.join(app.root_path,'static/surveys')
		if survey_source == "conv":
			survey_path = os.path.join(app.root_path,'static/conv_surveys')

		survey_dict = ""
		with open(survey_path+"/"+survey_file, 'r', encoding='utf-8') as f:
			survey_dict = json.load(f)

		json_resp = json.dumps({'status': 'OK', 'message':'', 'survey_data':survey_dict})
	else:
		json_resp = json.dumps({'status': 'ERROR', 'message':'Missing arguments'})

	return make_response(json_resp, 200, {"content_type":"application/json"})


@app.route('/er_bot')
def er_bot():
	return render_template('er_bot.html')

@app.route('/er_bot_get_conversation')
def er_bot_get_conversation():
	event_list = []
	logging.info("Viewing single conversation...")

	conv_id = request.args.get('conv_id')
	if conv_id is not None:
		logging.info("Conversation ID is not empty: "+str(conv_id))
		events = EventLog.query.filter_by(conv_id=conv_id).order_by(sqlalchemy.asc(EventLog.timestamp)).limit(1000)
		for event in events:
			logging.info("Event:"+str(event.event))
			event_list.append({'server_time':event.timestamp.isoformat(' '), 'data':json.loads(event.event)})

	json_resp = json.dumps(event_list)

	return make_response(json_resp, 200, {"content_type":"application/json"})


@app.route('/er_bot_conversations')
def list_er_bot_conversations():
	er_conversations = []
	logging.info("Rendering ER conversations...")

	conv_ids = {}

	allEvents = EventLog.query.order_by(sqlalchemy.desc(EventLog.timestamp)).limit(10000)
	for event in allEvents:
		logging.info("Conv ID:"+event.conv_id)
		if event.conv_id not in conv_ids:
			logging.info("New conversation ID, get first date:")
			no_events = EventLog.query.filter_by(conv_id=event.conv_id).order_by(sqlalchemy.asc(EventLog.timestamp)).count()
			logging.info("Number of events: "+str(no_events))
			oldest_event = EventLog.query.filter_by(conv_id=event.conv_id).order_by(sqlalchemy.asc(EventLog.timestamp)).first()
			if oldest_event:
				logging.info("Got first event" + oldest_event.timestamp.isoformat(' '))
				conv_ids[event.conv_id] = { "start_date":oldest_event.timestamp.isoformat(' '), "no_events": no_events }

	for key, value in conv_ids.items():
		er_conversations.append( { 'datetime': value["start_date"],'filepath': key,'len': value['no_events'] } )


	return render_template('list_er_conversations.html',
		conversations = er_conversations
	)

@app.route('/logErEvent')
def log_er_event():
	logging.info("Got log ER event request...")
	json_resp = json.dumps({})
	conv_id = request.args.get('conv_id')
	data = request.args.get('data')
	logging.info("Conv_id: "+str(conv_id))
	logging.info("Data: "+str(data))
	if conv_id is not None and data is not None:
		logging.info("There is data in it!")

		event_data = json.dumps({})
		try:
			event_data = json.loads(data)
		except ValueError:
			logging.info("Can't parse event data as JSON")
			json_resp = json.dumps({'status':'error', 'message':'Provided event data not JSON according to Python3 json.loads'})
			return make_response(json_resp, 200, {"content_type":"application/json"})

		eventLog = EventLog(conv_id = conv_id, event = json.dumps(event_data))

		logging.info("Adding event to log...")
		db.session.merge(eventLog)
		db.session.commit()

		json_resp = json.dumps( {'status':'OK', 'conv_id':conv_id, 'event_entry':event_data })
	else:
		logging.info("No data in event request")
		json_resp = json.dumps({'status':'error', 'message':'No data in event log request!'})

	return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route('/ttsRequest')
def tts_request():
	logging.info("tts request...")
	json_resp = json.dumps({})
	text_call = request.args.get('text')
	voice_call = request.args.get('voice')
	lang_call = request.args.get('lang')
	speed_reduction_call = request.args.get('speed_reduction')
	if text_call is not None:

		key = "dff62f4138fa4314bcd1bf12c3e97602"
		url_token = 'https://api.cognitive.microsoft.com/sts/v1.0/issueToken'
		url_synth = 'https://speech.platform.bing.com/synthesize'

		voice = "Microsoft Server Speech Text to Speech Voice (en-US, ZiraRUS)"
		if voice_call is not None:
			voice = voice_call

		lang = "en-US"
		if lang_call is not None:
			lang = lang_call

		text = text_call

		speed_reduction = 0.0
		if speed_reduction_call is not None:
			speed_reduction = speed_reduction_call

		client_app_guid = 'e0e6613c7f7f4a5dbc06d5ad592895b4'
		instance_app_guid = '94eb5ccc71344c27ae7ab3fd2e572a52'
		app_name = 'Test_Speech_Gen'
		audio_filename = "static/audio/audio_"+datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")+".wav"

		options = {
			"http":{
			}
		}
		
		logging.info("-------REQUEST 1--------")
		headers = {	'Ocp-Apim-Subscription-Key': str(key),
					'Content-Length': "0"}

		r = requests.post(url_token, headers=headers)#data = {'key':'value'})
		logging.info("Status Code:"+str(r.status_code))
		logging.info("Resp headers:"+str(r.headers))

		auth_token = r.content
		logging.info("Auth Token:"+str(auth_token))


		#make the request for audio synthesis

		logging.info("-------REQUEST 2--------")
		#https://stackoverflow.com/questions/45247983/urllib-urlretrieve-with-custom-header

		data = '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '+ \
		'xmlns:mstts="http://www.w3.org/2001/mstts" '+ \
		'xml:lang="'+str(lang)+'">'+ \
		'<voice xml:lang="'+str(lang)+'" '+ \
		'name="'+str(voice)+'">'+ \
		'<prosody pitch="high" rate="-'+speed_reduction+'%">'+ \
		str(text)+ \
		'</prosody>' + \
		'</voice>'+ \
		'</speak>'

		token_base64 = base64.b64encode(auth_token).decode('ascii')

		headers = {	'Authorization': "Bearer "+auth_token.decode('ascii'),
					'Content-Type': "application/ssml+xml",
					'X-Microsoft-OutputFormat': "riff-8khz-8bit-mono-mulaw",
					'X-Search-AppId': client_app_guid,
					'X-Search-ClientID': instance_app_guid,
					'User-Agent': app_name,
					'Content-Length': str(len(data))}

		#r2 = requests.post(url_synth, headers=headers, data = json.dumps({'content':data}))
		req = requests.Request('POST',url_synth, headers=headers, data = data.encode('utf-8'))
		prepared = req.prepare()
		pretty_print_POST(prepared)

		s = requests.Session()
		resp = s.send(prepared)

		logging.info("Status Code:"+str(resp.status_code))
		logging.info("Resp headers:"+str(resp.headers))
		#print("Resp content:"+str(resp.content))

		audio_fullpath = os.path.join(app.root_path, audio_filename)
		logging.info("Audio full path:"+audio_fullpath)
		with open(audio_fullpath, 'wb') as fd:
			for chunk in resp.iter_content(chunk_size=128):
				fd.write(chunk)
	
		if resp.status_code == requests.codes.ok:
			logging.info("TTS request success!!!")
			json_resp = json.dumps({"audio_file":audio_filename})
		else:
			logging.warn("!!!TTS request failed")
	else:
		logging.warn("Error, text empty!!!")


	return make_response(json_resp, 200, {"content_type":"application/json"})
	#return "", 200, {'Content-Type': 'text/html; charset=utf-8'}
