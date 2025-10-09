from colorama import Fore, Style

def debug_style(text: str) -> str:
    '''Print text with [DEBUG] & red colour.'''

    text = f'{Fore.RED}[DEBUG] {text}{Style.RESET_ALL}'
    
    return text

