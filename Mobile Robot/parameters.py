"""
PARAMETERS.PY contains code to run the menu.
There are two classes:
    - param: object for a single parameter
    - parameters_obj: data structure to contain all the relevant parameters
"""
# https://www.geeksforgeeks.org/dropdown-menus-tkinter/
from tkinter import *

def menu_box(parameters_lst):
    """
    creates the menu box, adding all the parameters in parameters_lst to a dropdown menu.
    outputs root, without starting the mainloop() yet.
    """

    # Create object
    root = Tk()

    # Adjust size
    root.geometry("297x399")

    # Add background
    bg = PhotoImage(file="roger-the-musclebound-kangaroo-is-back-and-it-looks-like-hes-been-hitting-the-weights.png")

    # Resize image
    bg = bg.subsample(2, 2)

    # Store image in label attribute
    label1 = Label(root, image=bg)
    label1.image = bg
    label1.place(x=0, y=0)

    # initialise lists that will contain all the things that are shown in the menu
    options_lst = []
    values_lst = []
    comments_lst = []
    min_lst = []
    max_lst = []

    # iterate through the parameters and add them to the menu's data structures
    for parameter in parameters_lst:
        if parameter.displayInMenu: #boolean on wether or not to show the parameter in the menu
            # get the values
            name = parameter.name
            value = parameter.value
            minimum = parameter.minValue
            maximum = parameter.maxValue
            comment = parameter.comments
            # append them  to the data structures
            options_lst.append(name)
            values_lst.append(value)
            comments_lst.append(comment)
            min_lst.append(minimum)
            max_lst.append(maximum)

    # variable strings for the menu
    clicked_var = StringVar()
    comments_var = StringVar()
    value_var = StringVar()
    minimum_var = StringVar()
    maximum_var = StringVar()

    # initial menu text
    clicked_var.set(options_lst[0])
    comments_var.set(comments_lst[0])
    value_var.set(values_lst[0])
    minimum_var.set('min: ' + str(min_lst[0]))
    maximum_var.set('max: ' + str(max_lst[0]))

    def optionSelected(smth):
        """
        Handler for when the user selects an option from the dropdown menu
        -> update the variable strings on the menu to their appropriate value
        """
        clicked_str = clicked_var.get()
        item_index = options_lst.index(clicked_str)
        comments_var.set(comments_lst[item_index])
        value_var.set(values_lst[item_index])
        bottomText.config(text="current: "+str(values_lst[item_index]))
        minimum_var.set('min: ' + str(min_lst[item_index]))
        maximum_var.set('max: ' + str(max_lst[item_index]))

    # Create Dropdown menu
    drop = OptionMenu(root, clicked_var, *options_lst, command=optionSelected)
    drop.pack()

    # boxes to thow info about currently selected parameter
    comments_box = Label(root, textvariable=comments_var)
    min_box = Label(root, textvariable=minimum_var)
    max_box = Label(root, textvariable=maximum_var)

    # box where the user can input values
    value_box = Entry(root, textvariable=value_var)

    # put boxes in the display
    comments_box.pack()
    min_box.pack()
    max_box.pack()
    value_box.pack()

    def commitInput():
        """
        Handler for when the user presses commit:
        store the proposed value, if it is valid
        """
        # get proposed value
        inp = value_var.get()

        # determine which parameter we're updating
        clicked_str = clicked_var.get()
        item_index = options_lst.index(clicked_str)

        # get appropriate min and max
        inp_max = max_lst[item_index]
        inp_min = min_lst[item_index]

        try:
            # crashes when str cannot be converted to int
            value_int = int(inp)

            # check if the value is within bounds
            if value_int <= inp_max and value_int >= inp_min:
                # update the parameter object
                param = parameters_lst[item_index]
                param.value = value_int

                # update the menu
                values_lst[item_index] = value_int

                bottomText.config(text="Comitted Input: " + inp)
            else:
                print('proposed value out of range')
        except:
            print('can only use ints')

    # make commit button and add to the screen
    printButton = Button(root,
                            text="Commit",
                            command=commitInput)
    printButton.pack()

    # Label Creation
    bottomText = Label(root, text="")
    bottomText.config(text="current: " + str(values_lst[0]))
    bottomText.pack()

    return root

class param:
    """
    param class: to define a single parameter. A parameter has the following attributes:
    - name: a string with the parameters name, to be used in the menu
    - value: an integer value with the parameters current value
    - displayInMenu: boolean on wether or not to show the parameter in the menu
    - minvalue: an integer, serves as a minimum for value
    - maxvalue: an integer, serves as a maximum for value
    - comments: a string, it is displayed in the menu and serves as comments to explain to the user how to use the variable
    """
    def __init__(self, name='------------', value=0, displayInMenu=False, minValue=0, maxValue=999999, comments=''):
        self.name = name
        self.value = value
        self.displayInMenu = displayInMenu
        self.minValue = minValue
        self.maxValue = maxValue
        self.comments = comments

class parameters_obj:
    """
    parameters_obj class: a parameters object holds all the individual parameters,
    so we can pass and access all the parameters by passing this single object.
    """

    def __init__(self):
        # param object : (name, int value, boolean to include in menu, min, max, comments)
        # warping parameters
        self.warp = param('WARP', 1, True, 0, 1, '"to warp or not to warp" - shakespeare, probably')
        self.initialiseWarp = param('initialise warp', 0, True, 0, 1, 'Do you want to initialise a custom warp ?')
        self.firstLoop = param('firstLoop', 1, True, 0, 1, 'Reset the warp shape, simply press commit')
        self.warpType = param('warpType', 1, True, 0, 1, '0 for top stretch, 1 for bottom shrink method')
        self.break1 = param(displayInMenu=True)

        # hough parameters
        self.useHough = param('USEHOUGH', 1, True, 0, 1, "halolo tuur")
        self.thresholdOrCanny = param('Thresh vs Canny', 1, True, 0, 1, '1 for Threshold, 0 for canny')

        self.cannyTresh = param('cannyTresh', 100, True, 0, 255, 'treshold for the canny')
        self.edgeTreshold = param('edgeTreshold', 45, True, 0, 255, "treshold for edge detection, to be divided by 1000")
        self.break3 = param(displayInMenu=True)

        self.dilationOrErosion = param('Dilation vs Erosion', 1, True, 0, 1, '1 for Erosion, 0 for Dilation')
        self.kernelmorph = param('kernelmorph', 1, True, 1, 5, 'Kernel for morphological operation')
        self.morphIterations = param('Morph_iterations', 1, True, 0, 5, 'Amount of morphological transformations')
        self.break4 = param(displayInMenu=True)

        self.houghthresh = param('Hough threshold', 55, True)
        self.minLineLength = param('minLineLength', 15, True)
        self.maxLineGap = param('maxLineGap', 10, True)
        self.maxNbLines = param('maxNbLines', 20, True)
        self.break2 = param(displayInMenu=True)

        # camera parameters
        self.autoExposure = param('AUTOEXPOSURE',0, True, 0, 1, "enable/disable auto exposure")
        self.exposure_time = param('exposure time', 6000, True)
        self.exposure_gain = param('exposure gain', 2, True)
        self.param3 = param(displayInMenu=True)

        # map parameters
        self.makeMap = param('MAKEMAP', 0, True, 0, 1, "enable/disable graph algorithm")
        self.mapHeight = param('map HEIGHT', 150, True, 0, 300, "height of the map border")
        self.mapTopWidth = param('map TOP WIDTH', 300, True, 0, 300, "width of the top map border")
        self.mapBottomWidth = param('map BOTTOM WIDTH', 100, True, 0, 300, "width of the bottom map border")

        # secondary data structure, to be used by the menu to iterate over the attributes
        self.parameters_lst = []


    def add_param(self):
        """
        automatically adds and checks all the attributes of self to its attribute list structure
        """
        attributes = self.__dict__
        for parameter_str in attributes:
            parameter = attributes[parameter_str]
            if parameter != self.parameters_lst:
                self.parameters_lst.append(parameter)
                print(parameter.name)
                assert type(parameter.name == str) # Variable - name ; should be a string
                assert type(parameter.value == int) # Variable - value ; should be an int -> implement float for treshold??
                assert type(parameter.displayInMenu == bool) # Variable - display in menu? ; should be a boolean
                assert type(parameter.minValue == int) # Variable - minimum value ; should be an int -> implement float for treshold??
                assert type(parameter.maxValue == int) # Variable - maximum value ; should be an int -> implement float for treshold??
                assert type(parameter.comments == str) # Variable - comments ; for example: 'only use uneven ints' and some description of what the variable does

# can be used to test the menu outside of a thread
testParameters = False
if testParameters:
    parameters = parameters_obj()
    parameters.add_param()
    root = menu_box(parameters.parameters_lst)
    root.mainloop()