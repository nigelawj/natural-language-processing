# Imports
import pytest
from unittest.mock import MagicMock, patch
import time
import logging

import sys
# This line prevents a __pycache__ folder from appearing in src folder; neater this way
sys.dont_write_bytecode = True
# NOTE: This is probably due to how pytest works: presumably tagger_test.py is imported in the testing, and since tagger_test.py imports
# tagger.py, it will cause a __pycache__ to appear for tagger.py; if tagger_test.py is executed without being imported, this will not occur

sys.path.insert(1, '../') # Allow importing of a module not in current dir

import tagger # tagger.py

# Define parameter defaults for testing
stopwordsPath = './stopwords-test.txt'
mockEsIndex = 'documents'
mockNodes = ['127.0.9.1:9200']
mockLogger = logging.getLogger()

workStartAM = 11
workEndHr = 16

max_features = 1000
max_iter = 1000
learning_offset = 50
top_words = 10

b = 2
o = 0

doc = "1 CEO’S SPEECH FOR DSO SCHOLARSHIP AWARD CEREMONY 2017 Good evening everyone, welcome to DSO; and to the DSO Scholarship Award Ceremony for 2017. DSO’s mission is to undertake defence research and development so that we can provide secret edge systems and solutions to meet Singapore’s defence and national security needs. We develop capabilities to solve operational problems. Less obvious but equally important is that we also need to assist Ministry of Defence (or MINDEF) and the Singapore Armed Forces (or SAF) to be future- ready; and ensure that they are not get caught off-guard by technological developments. To succeed, DSO must be at the leading edge of technology. DSO’s key asset is its people. Our 1,500 R&D staff possess technical expertise spanning many domains and disciplines relevant to the SAF. DSO invests significant resources to strengthen our technological base. We work very hard to ensure that we get our fair share of top Singaporean talent; and ensure that we provide our engineers and scientists with ample opportunities to update and upgrade themselves through courses and programs within DSO and outside DSO. This includes sponsoring deserving in-service staff who have the passion and potential on undergraduate and postgraduate scholarships (Masters and PhD) in areas relevant to defence. Since our corporatisation in 1997, DSO has sponsored more than 200 staff on scholarship to local and overseas universities. Today, I am very pleased to award scholarship to another 13 staff. 2 In previous years, the DSO Scholarship has been given out at a ceremony together with the DSTA Undergraduate Scholarship. This year, MINDEF decided to organise a combined ceremony for various undergraduate scholarship awards. Hence, this is the first time we organised DSO Scholarship Award Ceremony on our own. And there is no other better place to hold it other than our own home here at the new DSO Complex, and together with the DSO family. For those family members who were not here at our Open House in January this year, this is an opportunity for you to appreciate the work environment of your loved ones. As our scholars embark on their new adventure across the globe, I like to encourage them to continue being great ambassadors of DSO. You will not only be expanding your knowledge, but also your network of linkages, both professional and personal. Seize the opportunity to make this an exciting and fulfilling one. And it will be bonus if you spot and convince a Singaporean talent to join DSO! Finally, I like to remind our scholars of the strong network they have in DSO, and also in Singapore. Our colleagues, families and friends are rooting for you. And DSO looks forward to welcoming you come back to push the boundary and break new ground with your new knowledge and experience. So my heartiest congratulations to our recipients and their families once again. I wish you all the best in your endeavours ahead."

# DEFINE TESTS
def test_check_time():
   if (time.localtime().tm_hour in range(workStartAM, workEndHr)):
      with pytest.raises(SystemExit) as pytest_wrapped_e:
         tagger.check_time(workStartAM, workEndHr, mockLogger)

      assert pytest_wrapped_e.type == SystemExit
      assert pytest_wrapped_e.value.code == 5

@patch('tagger.Elasticsearch')
def test_connectDB(mock_es_connection):
	# Mock an ES object returned upon successful connection
	# NOTE: 'es = elasticsearch.Elasticsearch()' will always return a elasticsearch.client.Elasticsearch class
	#				regardless of ES connection success/fail; 
	# 			Code is structured to return the es object only if an es.count() request succeeded. More info: tagger.connectDB
	
	#	Since es.count() would only work if es connection was established, we test for a document count
	# Mock the es object + mock the es.count() method to a hardcoded result
	temp_mock_es = mock_es_connection.return_value
	temp_mock_es.count.return_value = {'count': 420}

	mock_es = tagger.connectDB(mockEsIndex, mockNodes, mockLogger)

	# On a successful connect, es.count() should be executed once, and output 420 documents (this will never fail actually :o)
	mock_es.count.assert_called_once()
	assert mock_es.count()['count'] == 420

def test_load_stop_words():
	stop_words = tagger.load_stop_words(stopwordsPath)

	# If successfully loaded, stopwords should be a list of len > 0
	assert isinstance(stop_words, list)
	assert len(stop_words) > 0

def test_preprocess_text():
	stop_words = tagger.load_stop_words(stopwordsPath)
	processed_doc = tagger.preprocess_text(doc, stop_words)

	# If successful preprocess, result should be a string smaller than original document
	assert isinstance(processed_doc, str)
	assert len(processed_doc) < len(doc)

def test_perform_LDA():
	stop_words = tagger.load_stop_words(stopwordsPath)
	processed_doc = tagger.preprocess_text(doc, stop_words)

	# Should return a list of tags - can be 0 tags
	assert isinstance(tagger.perform_LDA(processed_doc, max_features, max_iter, learning_offset, top_words), list)

# Target which method to mock: i.e. the es connection method - elasticsearch.Elasticsearch()
@patch('tagger.Elasticsearch')
def test_execute_es_query(mock_es_connection):
	# Define hardcoded es object + hardcoded msearch method to return the method params: which should be mock_req
	temp_mock_es = mock_es_connection.return_value
	temp_mock_es.msearch.return_value = {'response': 200}

	mock_es = tagger.connectDB(mockEsIndex, mockNodes, mockLogger)

	mock_res = tagger.execute_es_query(mock_es, mockEsIndex, b, o)
	
   # NOTE: assert_called_once_with() evaluates the variables then compares with original function call;
   # i.e. any variables like b and o are treated as their values respectively, and then compared with the original call as per mock test
   #
   # e.g. For the above statement, mock_res = tagger.execute_es_query(mock_es, b, o); where b = 2 and o = 0
   #      Hence assert_called_once_with() will check if what was passed in matches what the original tagger.py would do:
   #      will the body parameter contain 'size': 1 and o: 0?
   #
   # In simpler terms: if original code uses int(b/2), in assert_called_once_with() you do not have to strictly use int(b/2) for the checking,
   #                   you can just pass in int(2/2) since b = 2, it will be evaluated to 'size': 1; See below where size is int(b/2) and int(1)
	mock_es.msearch.assert_called_once_with(body=[{
      'index': 'documents'
   }, {
      'size': int(b/2),
      'query': {
         'bool': {
            'should': [{
               'range': {
                  'lastTagged': {
                     'lt': o
                  }
               }
            }, {
               'bool': {
                  'must_not': {
                     'exists': {
                        'field': 'tags'
                     }
                  }
               }
            }]
         }
      }, 
      'sort': [{
         'lastIndexed': {
            'order': 'desc'
         }
      }],
      '_source': 'content'
   }, {
      'index': 'documents'
   }, {
      'size': int(1), # no difference
      'query': {
         'bool': {
            'should': [{
               'range': {
                  'lastTagged': {
                     'lt': o
                  }
               }
            }, {
               'bool': {
                  'must_not': {
                     'exists': {
                        'field': 'tags'
                     }
                  }
               }
            }]
         }
      }, 
      'sort': [{
         'lastIndexed': {
            'order': 'asc'
         }
      }],
      '_source': 'content'
   }])

	assert mock_res['response'] == 200

@patch('tagger.Elasticsearch') # not mocking any return values but still need this to mock ES DB connection
@patch('tagger.bulk')
def test_get_actions_and_bulk(mock_bulk, mock_es_connection):
   mock_bulk.return_value = (b, [])
   mock_es = tagger.connectDB(mockEsIndex, mockNodes, mockLogger)

   # Typical es.msearch(es, b=2, o=round(time.time())) successful response, only content from _source is returned
   # NOTE: in these responses, the documents have tags already. This is OK. Existing tags will simply be replaced by the LDA's output
   #       If the LDA algorithm is unchanged, the new tags will be identical to old tags. Nonetheless they will still be replaced
   mock_res = {
      'responses': [{
         'took': 83,
         'timed_out': False,
         '_shards': {
            'total': 30, 
            'successful': 30, 
            'skipped': 0, 
            'failed': 0
         },
         'hits': {
            'total': 80,
            'max_score': None,
            'hits': [{
               '_index': 
               'documents',
               '_type': '_doc',
               '_id': 'https___hub_dso_org_sg_ceoblog_documents_ceo_dso_speech_at_star_launch_final_pdf_94',
               '_score': None,
               '_source': {
                  'content': '1 CEO DSO’S SPEECH AT LAUNCH OF THE SATELLITE TECHNOLOGY AND RESEARCH CENTRE ON 25 JANUARY 2018 Professor Ho Teck Hua, Senior Deputy President and Provost, National University of Singapore Mr. Beh Kian Teik, Assistant Managing Director and Chief Digital Officer, EDB Mr. Tang Kum Chuen, Deputy President (Corporate Development) and President (Satellite Systems), ST Electronics Mr. Jarry Herve, Chief Technical Officer, Thales Solutions Asia Ladies and Gentlemen, A very good morning. I am delighted to be here today for the launch of the Satellite Technology And Research Centre (or STAR Centre in short). DSO’s Journey in Space As a national R&D organisation, DSO invests in nascent technologies that could potentially become mainstream some 15 to 20 years later. Back in the late 1990’s, we identified satellite technology as an area of enormous promise. Apart from the value it could bring to Singapore, satellites are an exciting frontier, and the pinnacle of engineering. It is an area that excites and inspires. 2 In 2001, DSO partnered the Nanyang Technological University (or NTU) to form Centre for Research in Satellite Technologies (or CREST), subsequently renamed Satellite Research Center (or SaRC). Together, we set our sights on developing a micro-satellite. This first foray into space had (more than) its fair share of challenges. But we persevered and prevailed. In 2011, the X-SAT was successfully launched. Singapore had entered space – with its own indigenous development. The X-SAT proved to be a valuable tool for remote sensing, particularly for environmental monitoring purposes. It created excitement among technologists in the local ecosystem. Our journey in space reached new heights with the successful launch in 2015 of six locally built satellites, including micro-satellite payloads from NTU and NUS; and the first commercial electro-optical mini-satellite from ST Electronics – theTeLEOS-1. TeLEOS-1 is a Near Equatorial Orbit Earth Observation satellite. It provides insightful, relevant, and timely geo-spatial solutions for Maritime Security and Safety, Humanitarian Aid and Disaster Relief, Forestation and Agriculture, and Infrastructure Construction Progress, to name a few. TeLEOS-1 is the result of collaboration among our key technology players via the Joint Venture between DSO, NTU, and ST Electronics (Satcom & Sensor Systems). ST Electronics led 3 the design, development, system integration, and testing of the satellite while the joint team from DSO and CREST at NTU provided the systems engineering expertise to ensure transition from research to industrialisation. Additionally, CRISP at NUS assisted the project in image processing. Local satellite development benefitted from EDB’s support. EDB sees potential in the satellite sector. It believed in the local ecosystem and provided funding for some of the payloads in the 2015 six-in-one satellite launch. The success of TeLEOS-1 underscored the technical capability of the local ecosystem. It gave us the confidence to think where this could lead us. Located near the equator, Singapore is ideally positioned to develop and operate low inclination angle satellites to provide high temporal remote sensing and geospatial solutions for environmental monitoring, maritime situation awareness, and disaster management. We envision that if we could leverage our natural geographic advantage and harness emergent nanosatellite constellation technology, we could develop low-cost satellites in numbers to yield re-visit rates that are ideal for near real-time applications. EDB and ST Electronics (Satcom & Sensor Systems) share the same vision. Apart from economic value, the idea fits well into the overall vision for developing the satellite sector in Singapore. 4 5 STAR in Space The focal point of research to support the nanosatellite constellation idea is STAR Centre. STAR was formed under the ambit of an MOU between NUS and DSO back in February 2017 with the mission to be a leader in advanced distributed small satellite systems. It will develop cutting-edge capabilities in distributed satellite systems. STAR researchers worked in a laboratory at the NUS Faculty of Engineering for several months and moved to their permanent premises at SWIFT in November. I am pleased that STAR has a place to call home. I am optimistic that it would conduct outstanding research and fulfil the role. STAR is the latest in the longstanding collaboration between NUS and DSO. This collaboration is mutually beneficial, enabling DSO to tap the scientific talent in NUS to develop advanced technologies in areas relevant to defence. It has also expanded the technological base in Singapore; and invigorated interest in Science, Technology, Engineering and Mathematics (or STEM) among new cohorts of young Singaporeans. STAR will no doubt further reinforce this cooperation. 6 Conclusion STAR was born through the unwavering support and dedication of the team at NUS and my fellow colleagues. But it would not have been possible too without the belief and encouragement of EDB. I would like to take this opportunity to thank EDB for supporting us with the necessary resources to develop our team and fund our research in this area and for helping us construct valuable linkages with other research industries. With that, I wish STAR success in achieving its vision. Thank you. Word count: 790',
               },
               'sort': [1585204676000]
            }]
         },
         'status': 200
      }, {
         'took': 82,
         'timed_out': False,
         '_shards': {
            'total': 30,
            'successful': 30, 
            'skipped': 0, 
            'failed': 0
         },
         'hits': {
            'total': 80,
            'max_score': None,
            'hits': [{
               '_index': 'documents',
               '_type': '_doc',
               '_id': 'https___serene_beach_06053_herokuapp_com__822',
               '_score': None,
               '_source': {
                  'content': 'Web site created using create-react-appYou need to enable JavaScript to run this app.'
               },
               'sort': [1583816779000]
            }]
         },
         'status': 200
      }]
   }

   stop_words = tagger.load_stop_words(stopwordsPath)

   mock_actions = [j for j in tagger.get_actions(mock_res['responses'], mockEsIndex, mockLogger, stop_words, max_features, max_iter, learning_offset, top_words)]

   # get_actions() inserts a lastTagged value using round(time.time()) which matching values are unimportant, we change them to 420
   # to avoid errors
   for i in mock_actions:
      i['_source']['doc']['lastTagged'] = 420

   # Simulated LDA output: these are the tags obtained from LDA performed on these 2 documents in mock response
   # b=2: two responses (first and second half) with 1 document in each half
   mock_tags = []
   for res in mock_res['responses']:
      mock_doc_id = res['hits']['hits'][0]['_id']
      mock_doc = res['hits']['hits'][0]['_source']['content']
      mock_processed_text = tagger.preprocess_text(mock_doc, stop_words)
      mock_tags.append(tagger.perform_LDA(mock_processed_text, max_features, max_iter, learning_offset, top_words)[0])

   print(mock_tags)

   assert mock_actions == [{
      '_op_type': 'update', 
      '_index': 'documents', 
      '_type': '_doc', 
      '_id': 'https___hub_dso_org_sg_ceoblog_documents_ceo_dso_speech_at_star_launch_final_pdf_94', 
      '_source': {
         'doc': {
            'tags': mock_tags[0], 
            'lastTagged': 420
         }
      }
   }, {
      '_op_type': 'update', 
      '_index': 'documents', 
      '_type': '_doc', '_id': 
      'https___serene_beach_06053_herokuapp_com__822', 
      '_source': {
         'doc': {
            'tags': mock_tags[1], 
            'lastTagged': 420
         }
      }
   }]

   assert mock_bulk(mock_es, mock_actions) == (len(mock_res['responses']), [])
	