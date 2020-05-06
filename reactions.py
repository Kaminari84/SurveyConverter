#More surveys:
# - https://www.surveymonkey.com/mp/university-student-satisfaction-survey-template/?ut_source=mp&ut_source2=sample-survey-questionnaire-templates&ut_source3=survey-template-gallery
# - https://www.researchgate.net/post/Does_a_comprehensive_list_of_validated_surveys_questionnaires_and_instruments_exist_anywhere_on_the_web
# - https://www.rand.org/health-care/projects/acove/survey.html
# - https://people.umass.edu/aizen/pdf/tpb.measurement.pdf
# - https://www.zoho.com/survey/templates/marketing/customer-expectations-survey.html
# - https://www.valuescentre.com/tools-assessments/pva/
# - https://survey.valuescentre.com/survey.html?id=V12S6O5DPS-uqtP8MuPkM7aEnWbc2oNdnCAs8vqA-S90I1_r6KnOng
# - https://webspace.ship.edu/cgboer/valuestest.html
reaction_repo = {
    # --- Introductions ---
    "intro_list": [
        "Hi, my name is {name}. I would like to talk to you about {topic}.",
        "Hi, I am {name}. Let's talk a bit about {topic}.",
        "Hi, I am {name}. I'd like to ask you a few questions about {topic}.",
        "Hi, my name is {name}. Let's start by talking about {topic}.",
        "Hey, I am {name}. Let's talk about {topic}."
    ],
    # --- Endings ----
    "close_list": [
        "We are done! Thanks.", 
        "We are done! Thanks for your patience.",
        "This is it! Thank you for your time.",
        "We've completed the whole survey! Thanks a lot!"
    ],
    # --- Communicating survey progress ---
    "progress_list": [
        "We are currently at question {d} out of {n}.",
        "We are done with {d} questions, still {l} to go.",
        "We have completed {percent}% of the survey.",
        "We are done with {percent}% of our questions."
    ],
    "progress_middle_list": [
        "We are now in the middle of the survey",
        "We are about half way through",
        "Half of the survey is done, thanks for your patience",
        "We're half way there, still {l} questions to go"
    ],
    "progress_end_list": [
        "We are almost done, thanks for your patience",
        "We're mostly done, just {l} questions left",
        "We are almost at the end, thank you for staying that long",
        "We are almost at the end of the survey. I appreciate your patience."
    ],
    # --- Reactions to various answers ---
    "reaction_positive_list": [
        "I am glad to hear that",
        #"I am happy this is not a problem",
        "I am happy that's the case",
        "Good to hear that",
        "That's good",
        "Sounds good",
        "Sounds nice",
        "That sounds positive",
        "Okay, that's good",
        "Great!",
        "That's really great!"
    ],
    "reaction_negative_list": [
        "That is frustrating",
        "I am sorry to hear that",
        "So sorry about that",
        "Thanks for sharing that",
        "That sounds stressful",
        "That's hard to hear"
    ],
    "reaction_neutral_list": [
        "Got it",
        "Thanks for sharing",
        "Got it! Thanks for sharing",
        "Sure",
        "Noted",
        "Thanks for letting me know",
        "Okay, I'm getting a better idea of your answers",
        "Thank you for your answer",
    ],
    # --- Survey questions augmentations ---
    #If I - pronoun in sent - problem - can't use
    #If q starts with verb - rem verb, but not always
    
    # --- Verb pefixes ---
    #to glue to: "Can you tell me" + "what gender do you identify as?"
    'q_verb_prefix': {
      "prefix": [
        "Can you tell me", 
        "Could you tell me",
        "Would you mind sharing",
        "Would you mind telling me",
        "Can I ask you"
      ],
      "conv": []
    },

    #to glue to: "Indicate your current age."
    'q_non_verb_prefix': {
      "prefix": [
        "Can you",
        "Could you please",
        "Would you please",
        "Would you be able to",
        "Can I ask you to"
      ], 
      "conv":[{"type":"replace", "from":".", "to":"?"}]
    },
                 
    #to glue to: "Feeling tired or having little energy."
    'q_add_question': {
      "prefix": [
        "Have you experienced",
        "Did you experience",
        "Please tell me if you have experienced",
        "Can you share whether you have experienced"
      ],
      "conv":[{"type":"replace", "from":".", "to":"?"}]
    },
                 
    #to glue to: "Are you married?"
    'q_prefix_if_you': {
      "prefix": [
        "Can you indicate whether",
        "Can you tell me whether",
        "Could I ask you whether",
        "Could you tell me whether",
        "Can you share if",
        "Can you tell me if",
        "Please indicate if",
        "Please tell me if",
        "Please tell me whether",
        "Please indicate whether"
      ],
      "conv":[{"type":"replace", "from":"are you", "to":"you are"}]
    },
                 
    #to go with: "This course requires us to understand concepts taught by the lecturer."
    'q_prefix_if_i': {
      "prefix": [
        "Would you say that",
        "Can you say that",
        "Please indicate the extent to which"
      ],
      "conv": [
        {"type":"replace", "from":"are you", "to":"you are"},
        {"type":"replace", "from":"i ", "to":"you "},
        {"type":"replace", "from":"I ", "to":"you "},
        {"type":"replace", "from":"am ", "to":"are "},
        {"type":"replace", "from":"we ", "to":"you "},
        {"type":"replace", "from":"us ", "to":"you "},
        {"type":"replace", "from":"my ", "to":"your "},
        {"type":"replace", "from":"have you", "to":"you have"},
      ]
    },
                 
    #to go with: "If you've had any days with issues above, how difficult have these problems..."
    'q_none':{"prefix":[],
        "conv":[]
    },

    # -- repeated prefix additions - glue to prefix in further questions ---
    "repeat_prefix_additions": [
        "Next",
        "Going forward",
        "Moving on",
        "Further",
        "Continuing"
    ],

    # --- Transitions between topics/sections ---
    "section_transitions": [
        "Next, I want to talk about {section_topic}",
        "Thank you fot your answers, let's move on tp talking about {section_topic}",
        "This is great, let's move on to questions about {section_topic}",
        "Thanks for all the answers so far, could we now talk abut {section_topic}"
    ]
    # --- Validations for specific types of answers (open-text) ---
    # --- Requests for elaboration / probing (open-text) ---
    # --- Clarifications (open-text) ---


}
