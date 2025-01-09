
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import os

current_folder_path = os.getcwd()
download_dir = current_folder_path + "\\IBOVDia"

def get_browser_driver():
    # Configuração para o Chrome
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")  # Rodar em modo headless
    chrome_options.add_argument("--disable-gpu")  # Opcional: desabilitar GPU, se necessário
    chrome_prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", chrome_prefs)
    
    # Configuração para o Firefox
    firefox_options = FirefoxOptions()
    firefox_options.add_argument("--headless")  # Rodar em modo headless
    firefox_options.set_preference("browser.download.folderList", 2)
    firefox_options.set_preference("browser.download.dir", download_dir)
    firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/csv")
    
    # Configuração para o Edge
    edge_options = EdgeOptions()
    edge_options.add_argument("--headless")  # Rodar em modo headless
    edge_prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    edge_options.add_experimental_option("prefs", edge_prefs)

    # Lógica para detectar o navegador (caso queira rodar de acordo com a preferência do usuário)
    browser_name = os.getenv("BROWSER", "chrome").lower()  # Padrão para Chrome, pode mudar via variável de ambiente

    if browser_name == "chrome":
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        print("Driver do Chrome iniciado com sucesso!")
        return driver
    
    elif browser_name == "firefox":
        driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
        print("Driver do Firefox iniciado com sucesso!")
        return driver
    
    elif browser_name == "edge":
        driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()), options=edge_options)
        print("Driver do Edge iniciado com sucesso!")
        return driver
    
    else:
        raise ValueError(f"Navegador {browser_name} não suportado. Por favor, escolha entre 'chrome', 'firefox' ou 'edge'.")
