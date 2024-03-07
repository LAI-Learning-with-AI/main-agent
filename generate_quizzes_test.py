# fix errors when importing locally versus as submodule
if __package__ is None or __package__ == '':
    from generate_quizzes import generate_quiz
else:
    from .generate_quizzes import generate_quiz

# generate quiz and return parsed info
# note: to see the raw, un-parsed quiz, change generate_quiz function to return "response"
print(generate_quiz(10, 'multiple choice', 'dimensionality reduction, supervised learning'))