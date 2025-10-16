from colorama import Fore, Style

def style_text(text: str, colour: str, msg: str | None = None) -> str:
    '''Print text with a chosen colour.'''
    clr_code = getattr(Fore, colour.upper())

    # eg [DEBUG]/[WARNING]
    if msg:
        return f'{clr_code}[{msg.upper()}] {text}{Style.RESET_ALL}'
    
    # No msg
    return f'{clr_code}{text}{Style.RESET_ALL}'


def debug_style(text: str) -> str:
    '''Print text with [DEBUG] & green colour.'''

    return style_text(text=text, colour='green', msg='debug')


def warning_style(text: str) -> str:
    '''Print text with [WARNING] & red colour.'''

    return style_text(text=text, colour='red', msg='warning')

