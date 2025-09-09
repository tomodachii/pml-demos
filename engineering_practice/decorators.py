# Functions as first-class citizens
from datetime import datetime


def increment(a):
    return a + 1


def doubling(a):
    return a * 2


def compose(f, g):
    return lambda x: g(f(x))


increment_then_doubling = compose(increment, doubling)

print(increment_then_doubling(2))


# Inner Functions
def parent():
    print("Printing from parent()")

    def first_child():
        print("Printing from first_child()")

    def second_child():
        print("Printing from second_child()")

    second_child()
    first_child()


parent()


# Functions as Return Values


# Simple Decorator
def not_during_the_night(func):
    def wrapper():
        if 7 <= datetime.now().hour < 22:
            func()
        else:
            print("not now")
    return wrapper


# Syntatic sugar
@not_during_the_night
def say_whee():
    print("Whee!")


# say_whee = not_during_the_night(say_whee)
say_whee()
