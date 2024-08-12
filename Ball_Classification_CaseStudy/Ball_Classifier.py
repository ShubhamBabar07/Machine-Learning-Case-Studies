# Ball Classification Case Study 

# Rough = 1
# Smooth = 0

# Tennis = 1
# Cricket = 2

from sklearn import tree

def BallClassfier(weight, surface):

    # feature encoding
    Features = [[35,1], [47,1], [90,0], [48,1], [90,0], [35,1], [92,0], [35,1], [35,1], [35,1]]

    # labels encoding
    Labels = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1 ]

    obj = tree.DecisionTreeClassifier()

    # train
    obj = obj.fit(Features, Labels)

    # test
    result = obj.predict([[weight, surface]])

    if result == 1:
        print("Your object looks like Tennis Ball")
    elif result == 2:
        print("Your object looks like Cricket Ball")


def main():

    print("---------------- Ball Classification Case Study ----------------")

    weight = int(input("Enter weight of object : "))

    surface = input("What is the surface type of Object, Rough or Smooth : ")

    if(surface.lower() == "rough"):
        surface = 1
    elif(surface.lower() == "smooth"):
        surface = 0
    else:
        print("Error : Wrong Input")
        exit()

    BallClassfier(weight, surface)


if __name__ == "__main__":
    main()