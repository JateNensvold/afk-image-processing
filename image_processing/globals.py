import os

DEBUG = True
# DEBUG = False


baseDir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)))
databasePath = os.path.join(baseDir, "database")
databaseHeroesPath = os.path.join(databasePath, "hero_icon/*jpg")

flannPath = os.path.join(databasePath, "baseHeroes.flann")
staminaTemplatesPath = os.path.join(baseDir, "stamina_templates")
levelTemplatesPath = os.path.join(baseDir, "level_templates")


siPath = os.path.join(baseDir, "si")
siTrainPath = os.path.join(siPath, "train")
siBasePath = os.path.join(siPath, "base")
fiPath = os.path.join(baseDir, "fi")

lvlPath = os.path.join(baseDir, "levels")
