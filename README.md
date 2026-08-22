# Ask Jaanvi

### NLP-Powered Conversational Portfolio Assistant

Ask Jaanvi is a conversational AI system built to make a developer
portfolio interactive through natural language.

Instead of relying only on static portfolio pages, users can ask
questions about Jaanvi's skills, projects, experience, engineering
approach, technical capabilities, and professional fit.

The system understands the user's query, identifies its intent,
retrieves relevant information from a structured portfolio knowledge
base, and uses an LLM to generate a contextual and grounded response.

Ask Jaanvi is built as a **separate AI backend service**, allowing it
to be integrated into an existing portfolio frontend through a REST API.

---

## 🌐 Live

### AI Backend API

https://ask-jaanvi.onrender.com/

### Streamlit Demo Frontend

https://ask-jaanvi-ke8xp4wkmsgmrxfjjhabk9.streamlit.app/
The Streamlit application is a demonstration interface for the AI
service.

The same backend can be integrated directly into the main portfolio
through its REST API.

---

# What is Ask Jaanvi?

Ask Jaanvi is an application of **NLP + Information Retrieval +
LLM-based generation** to a real-world portfolio use case.

A user can ask questions naturally instead of searching through
different sections of a resume or portfolio.

For example:


What projects has Jaanvi built?

What technologies does Jaanvi know?

Why should we hire Jaanvi?

Would Jaanvi be a good fit for a Junior AI Engineer role?

Jaanvi ko AI Engineer ke liye kyun hire karna chahiye?

How would Jaanvi approach a new AI problem? 

# Why This Project?

A traditional portfolio is mostly static.

The visitor reads:

Skills
Projects
Experience
Achievements

Ask Jaanvi changes the interaction model:

Static Portfolio
       ↓
Natural Language Question
       ↓
AI understands the question
       ↓
Relevant portfolio information is retrieved
       ↓
LLM generates the response
       ↓
User gets a conversational answer

The project explores how NLP and LLM systems can be used to transform
structured information into an interactive experience.

It also demonstrates an important AI engineering principle:

The LLM generates the response, but trusted portfolio information
comes from the application's knowledge layer.

# Core AI / NLP Pipeline

The complete request pipeline is:

                    USER QUERY
                        │
                        ▼
               ┌─────────────────┐
               │ Query Processing│
               │  Tokenization   │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Intent          │
               │ Classification  │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Knowledge       │
               │ Retrieval       │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Context         │
               │ Construction    │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ System Prompt + │
               │ Trusted Context │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Groq + Llama    │
               │ LLM Generation  │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Final Grounded  │
               │ Response        │
               └─────────────────┘
# How It Works
1. User Query

The user sends a natural-language question.

Example:

"humme Jaanvi ko AI engineer ke liye kyu hire karna chahiye?"

The query is sent to the FastAPI backend.

2.Intent Classification

The system determines what kind of request the user is making.

Current intent categories include:

identity_factual
jaanvi_factual
technical_reasoning
hybrid
off_topic

Examples:

"What projects has Jaanvi built?"
→ jaanvi_factual
"How would you build a spam classifier?"
→ technical_reasoning
"Can Jaanvi build a spam classifier?"
→ hybrid

This allows the system to handle portfolio questions differently from
general technical questions.

3. Knowledge Retrieval

Jaanvi's information is stored as structured knowledge.

The knowledge base contains areas such as:

Identity
Education
Skills
Projects
Experience
Achievements
Interests
Engineering Approach

The retriever searches the knowledge base for information relevant to
the user's question.

For example:

Query:
"jaanvi tech me kaisi h"

        ↓

Relevant Sections:
identity
skills

Instead of blindly passing the complete portfolio to the LLM, the
system retrieves relevant information first.

4. Context Construction

The retrieved information is converted into context that can be used
by the language model.

Conceptually:

User Question
      +
Intent
      +
Relevant Portfolio Facts
      +
Response Instructions
      ↓
LLM Context

This creates a grounded generation pipeline.

5. LLM Generation

The final context is sent to the Groq API using a Llama model.

The LLM is responsible for turning the structured context into a
natural conversational response.

The model is instructed to:

Answer the actual question
Use relevant retrieved information
Avoid inventing portfolio facts
Match the user's language
Distinguish verified experience from potential capability
Stay conversational rather than producing unnecessary reports
English + Hinglish Support

Ask Jaanvi adapts to the user's language.

For example:

User:
Why should we hire Jaanvi?

Assistant:
Jaanvi would be a strong fit because...

The same intent can be expressed in Hinglish:

User:
Humme Jaanvi ko AI Engineer ke liye kyun hire karna chahiye?

Assistant:
Jaanvi ko AI Engineer ke liye hire karna strong choice
ho sakta hai because...

The system is designed to keep Hinglish as natural Hinglish rather than
automatically converting it into formal Hindi.

Technical terms such as:

AI/ML
NLP
LLM
API
Python
FastAPI

remain in English where appropriate.

Hiring & Professional Questions

Ask Jaanvi is also designed to handle professional-fit questions.

Examples:

Why should we hire Jaanvi?

Would she be a good Junior AI Engineer?

What makes her suitable for an AI role?

What if the role requires a technology she hasn't listed?

For these questions, the system uses verified portfolio information
and makes a positive professional assessment without intentionally
claiming unsupported experience.

If a particular technology is not present in the knowledge base, the
assistant should not claim direct experience with it.

Instead, it can discuss relevant transferable skills, adaptability,
and learning ability.

Technical Reasoning

Ask Jaanvi is not restricted to portfolio questions.

It can also handle general AI/ML/software engineering questions.

Examples:

How would you build a spam classifier?

What is RAG?

How does a CNN work?

How would you deploy an ML model?

How would you design an AI-powered application?

For such questions, the system can use general technical reasoning
rather than forcing an answer from Jaanvi's portfolio.

This creates two complementary capabilities:

Portfolio Knowledge
        +
General Technical Reasoning
# System Architecture
                         MAIN PORTFOLIO
                               │
                               │
                         User Question
                               │
                               ▼
                  ┌────────────────────────┐
                  │    Ask Jaanvi API      │
                  │        FastAPI         │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │   Intent Classifier    │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │  Knowledge Retriever   │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │   Context Builder      │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │    Prompt Builder      │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │      Groq + Llama      │
                  │     LLM Generation     │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │   Grounded Response    │
                  └───────────┬────────────┘
                              │
                              ▼
                         MAIN PORTFOLIO
Frontend & Backend Architecture

Ask Jaanvi is intentionally separated into two layers.

┌──────────────────────────┐
│      Main Portfolio      │
│                          │
│  Portfolio UI            │
│  Chat UI                 │
└────────────┬─────────────┘
             │
             │ REST API
             ▼
┌──────────────────────────┐
│     Ask Jaanvi Backend   │
│                          │
│ FastAPI                  │
│ Intent Classification    │
│ Retrieval                │
│ Prompting                │
│ LLM Integration          │
└──────────────────────────┘

This separation means the AI logic does not need to be tightly coupled
to a particular frontend.

The API can be consumed by:

The main portfolio
Streamlit
Web applications
Mobile applications
Other frontend clients
API
POST /api/v1/chat
Request
{
  "message": "Why should we hire Jaanvi?",
  "conversation_history": []
}
Response
{
  "response": "Jaanvi would be a strong fit...",
  "intent": "jaanvi_factual",
  "confidence": 0.9
}
Production Backend

The FastAPI backend is deployed on Render:

https://ask-jaanvi.onrender.com/

API endpoint:

POST https://ask-jaanvi.onrender.com/api/v1/chat
Demo Frontend

A Streamlit interface is included to demonstrate and test the API.

https://ask-jaanvi-ke8xp4wkmsgmrxfjjhabk9.streamlit.app/

The Streamlit application is not the core AI system.

It acts as a client of the FastAPI service:

Streamlit
    │
    │ POST /api/v1/chat
    ▼
FastAPI
    │
    ▼
NLP + Retrieval + LLM
    │
    ▼
Response
    │
    ▼
Streamlit

The same API can be connected to the main portfolio.

Tech Stack
AI / NLP
Natural Language Processing
Intent Classification
Information Retrieval
Prompt Engineering
LLM Integration
Groq API
Llama LLM
Backend
Python
FastAPI
Uvicorn
Pydantic
REST API
Frontend / Demo
Streamlit
Python
Requests
Custom CSS
Testing
Pytest
Deployment
Render
Streamlit Community Cloud
Development
Git
GitHub
Environment Variables
Project Structure
Ask jaanvi/
│
├── app/
│   │
│   ├── api/
│   │   └── routes/
│   │       └── chat.py
│   │
│   ├── core/
│   │   └── intent.py
│   │
│   ├── knowledge/
│   │   ├── loader.py
│   │   └── retriever.py
│   │
│   ├── llm/
│   │   ├── client.py
│   │   └── prompt.py
│   │
│   └── main.py
│
├── tests/
│   ├── test_api.py
│   ├── test_intent.py
│   └── test_knowledge.py
│
├── streamlit_app.py
├── requirements.txt
├── .env
└── README.md
Request Flow

A complete request follows this flow:

1. User asks a question
          ↓
2. Frontend sends POST request
          ↓
3. FastAPI receives request
          ↓
4. Intent is classified
          ↓
5. Relevant knowledge is retrieved
          ↓
6. Context is constructed
          ↓
7. System prompt is selected
          ↓
8. Groq / Llama generates response
          ↓
9. API returns JSON
          ↓
10. Frontend displays response
Grounding & Reliability

The system separates factual portfolio information from LLM-generated
language.

The assistant is designed not to invent:

Projects
Skills
Employers
Achievements
Metrics
Responsibilities
Experience

For unsupported information, the assistant should acknowledge the
limitation rather than presenting an invented fact as verified
experience.

This makes the system more suitable for professional portfolio and
recruitment use cases.

Security

User messages are treated as untrusted input.

The system includes instructions designed to prevent user messages from
overriding system-level behavior.

The assistant is also designed to avoid:

Revealing private information
Fabricating portfolio information
Following prompt-injection instructions
Treating user-provided claims as verified portfolio facts

The Groq API key is stored server-side using environment variables and
is not exposed to the frontend.

Testing

The project includes automated tests for:

API behavior
Intent classification
Knowledge retrieval
Retrieval relevance
Empty knowledge-base behavior
Status preservation
Edge cases

Run:

pytest -q

Current test status:

162 passed
