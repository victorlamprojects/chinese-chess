# Global Config
from common.GlobalConfig import *
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from os import system, name, path
from bs4 import BeautifulSoup
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
import warnings
import time
import requests
warnings.filterwarnings("ignore", category=DeprecationWarning) 

## Chrome
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")
driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), chrome_options=chrome_options)

def extractGameBoard(table):
    gameboard = [["--"]*9 for _ in range(10)]
    cnt = -1
    for td in table.find_elements(By.TAG_NAME, "td"):
        if "<img src=" not in td.get_attribute("innerHTML"):
            continue
        imgName = td.find_element(By.TAG_NAME, "img").get_attribute("src")
        img = imgName.split("/")[-1][:2]
        cnt += 1
        if img == "fl":
            continue
        img = img.replace("w", "r")
        gameboard[cnt//9][cnt%9] = img
    return gameboard

# http://www.ztchess.com/xqgame/gview.asp?id=044341D731A1AC
def extractGameInfo(gameUrl, gameResult):
    driver.get(gameUrl)
    time.sleep(2)
    movecontent = driver.find_element(By.ID, "movecontent")
    df = pd.DataFrame(columns=[gameResult,"","","","","","","",""], )
    for move in movecontent.find_elements(By.TAG_NAME, "tr"):
        if "<a id=\"Move" not in move.get_attribute("innerHTML"):
            continue
        move.find_element(By.TAG_NAME, "a").click()
        time.sleep(0.05)
        gameboardtd = driver.find_element(By.ID, "gameboardtd")
        gameboard = extractGameBoard(gameboardtd)
        for row in gameboard:
            df.loc[len(df.index)] = row
    return df

domain = "http://www.ztchess.com"
for pageNum in range(2, 4):
    url = f"http://www.ztchess.com/xqdata/gamelist.asp?page={pageNum}"
    r = requests.get(url)
    soup = BeautifulSoup(r.content, features="html.parser")
    table = soup.body.find(id="bpwPlayergame")

    for tr in table.find_all("tr")[1:]:
        gameLink = tr.find_all("td")[4].find("a")
        gameResult = "d"
        if gameLink.find("font").get_text() == "胜":
            gameResult = "r"
        elif gameLink.find("font").get_text() == "负":
            gameResult = "b"
        gameId = gameLink.attrs['href'].split("=")[-1]
        gameUrl = f"http://www.ztchess.com/xqgame/gview.asp?id={gameId}"
        df = extractGameInfo(gameUrl, gameResult)
        df.to_csv(f"{data_folder_path}/{gameId}.csv", index=False)