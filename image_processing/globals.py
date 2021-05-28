import os

baseDir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)))
databasePath = os.path.join(baseDir, "database")
databaseHeroesPath = os.path.join(databasePath, "hero_icon/*jpg")

flannPath = os.path.join(databasePath, "baseHeroes.flann")
numbersPath = os.path.join(baseDir, "numbers")

siPath = os.path.join(baseDir, "si")
fiPath = os.path.join(baseDir, "fi")
