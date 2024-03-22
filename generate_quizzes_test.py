# fix errors when importing locally versus as submodule
if __package__ is None or __package__ == '':
    from generate_quizzes import generate_quiz
else:
    from .generate_quizzes import generate_quiz

json = generate_quiz(10, 'MULTIPLE_CHOICE, SHORT_ANSWER', 'dimensionality reduction, ensemble learning', True)
print(json)