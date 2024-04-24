# Tutor Agent using Retrieval Augmented Generation (RAG)

A backend for a conversational tutor bot with functionality for generating quizzes, providing helpful resources, and tracking learning progress. 
Developed as a part of a larger senior project at the University of Florida.


## Getting Started

```
git clone https://github.com/LAI-Learning-with-AI/main_agent
cd main_agent
pip install -r requirements.txt
```

- Copy utils/.env.example -> utils/.env
- Fill out details of .env with OpenAI API key and Postrgesql Database Password 
  - (Note: collection name must be "corpus")

## Acknowledgments

Originally based on the learning-agent repository: 
- [EpicGazel/learning-agent](https://github.com/EpicGazel/learning-agent)

Developers for this repo:
- Lamb, Joshua (joshual55)
- Lewis, Zane (EpicGazel)

Thank you to Dr. Catia Silva for advising and overseeing the project.