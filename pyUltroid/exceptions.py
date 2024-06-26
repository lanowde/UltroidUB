# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

"""
Exceptions which can be raised by py-Ultroid Itself.
"""


class pyUltroidError(Exception):
    pass


class TelethonMissingError(ImportError):
    pass


class DependencyMissingError(ImportError):
    pass


class UploadError(Exception):
    pass


class DownloadError(Exception):
    pass
