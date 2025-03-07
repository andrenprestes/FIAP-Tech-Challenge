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
from pathlib import Path
import os

download_dir = Path("IBOVdia").resolve()
download_dir.mkdir(exist_ok=True)

print(f"Diretório de download configurado: {download_dir}")

def get_browser_driver():
    """
    Configures and returns a WebDriver instance for the desired browser.

    The browser is determined by the `BROWSER` environment variable, which can be set to 
    "chrome", "firefox", or "edge". If no value is set, "chrome" is used as the default.

    Each browser driver is configured to run in headless mode and sets up a default 
    download directory. The download directory is defined as a subfolder named 
    "IBOVDia" in the current working directory.

    Returns:
        selenium.webdriver.Chrome or selenium.webdriver.Firefox or selenium.webdriver.Edge: 
            The configured WebDriver instance for the specified browser.

    Raises:
        ValueError: If the specified browser in the `BROWSER` environment variable is not supported.

    Notes:
        - For Chrome and Edge, the download preferences include:
          * Default download directory set to `download_dir`.
          * Automatic download without prompt.
          * Safe browsing enabled.
        - For Firefox, preferences include:
          * Default download directory set to `download_dir`.
          * Automatic handling of CSV downloads without prompt.
    """
    # Configuração para o Chrome
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")  # Rodar em modo headless
    chrome_options.add_argument("--disable-gpu")  # Opcional: desabilitar GPU, se necessário
    chrome_prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", chrome_prefs)
    
    # Configuração para o Firefox
    firefox_options = FirefoxOptions()
    firefox_options.add_argument("--headless")  # Rodar em modo headless
    firefox_options.set_preference("browser.download.folderList", 2)
    firefox_options.set_preference("browser.download.dir", str(download_dir))
    firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/csv")
    
    # Configuração para o Edge
    edge_options = EdgeOptions()
    edge_options.add_argument("--headless")  # Rodar em modo headless
    edge_prefs = {
        "download.default_directory": str(download_dir),
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
        firefox_options.binary_location = "/usr/bin/firefox"
        driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
        print("Driver do Firefox iniciado com sucesso!")
        return driver
    
    elif browser_name == "edge":
        driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()), options=edge_options)
        print("Driver do Edge iniciado com sucesso!")
        return driver
    
    else:
        raise ValueError(f"Navegador {browser_name} não suportado. Por favor, escolha entre 'chrome', 'firefox' ou 'edge'.")
