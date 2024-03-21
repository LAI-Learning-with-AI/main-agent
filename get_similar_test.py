# fix errors when importing locally versus as submodule
if __package__ is None or __package__ == '':
    from get_similar import get_similar
else:
    from .get_similar import get_similar


def test_get_similar():
    print(get_similar(["Backpropagation", "Reinforcement Learning"]))


def run_tests():
    print("Running get_similar test...")
    test_get_similar()


if __name__ == '__main__':
    run_tests()
