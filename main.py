from src.question1 import question1a, question1b, question1c
from src.question2 import question_2a, question_2b, question_2c, question_2c_raster
from src.question3 import question3b


def question1():
	print("Question 1")
	question1a()
	question1b()
	question1c()


def question2():
	print("Question 2")
	question_2a(resolution=25, show=False)
	question_2b(show=False)
	question_2c(resolution=25, show=False)
	question_2c_raster(resolution=25, show=False)


def question3():
	print("Question 3")
	question3b()


if __name__ == '__main__':
	question1()
	question2()
	question3()

