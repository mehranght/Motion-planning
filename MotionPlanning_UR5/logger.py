"""
Copyright Apera AI Inc. 2021
"""
import logging
import logging.config
from pathlib import Path


def set_logging_config(filename, console):
    config = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '%(asctime)s - %(filename)s:%(lineno)d - %(message)s',
                'datefmt': '%m/%d %H:%M:%S'
            }
        },
        'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'filename': str(filename)
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['file'],
            'propagate': False
        }
    }

    if console:
        config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        }
        config['root']['handlers'].append('console')

    logging.config.dictConfig(config)


def get_current_logfile():
    logger = logging.getLogger()
    handler = logger.handlers[0]
    return Path(handler.baseFilename)
