# fix errors when importing locally versus as submodule
if __package__ is None or __package__ == '':
    from generate_quizzes import generate_quiz
else:
    from .generate_quizzes import generate_quiz

print(generate_quiz(5, 'multiple choice', 'dimensionality reduction, supervised learning'))