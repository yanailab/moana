# Copyright (c) 2018 NYU
#
# This file is part of Moana.

"""Functions for interacting with the public Ensembl FTP server."""

import ftplib
import re
from collections import OrderedDict
import logging

from typing import Dict

_LOGGER = logging.getLogger(__name__)


def get_latest_release(ftp: ftplib.FTP = None) -> int:
    """Use files on the Ensembl FTP server to determine the latest release.

    Parameters
    ----------
    ftp : ftplib.FTP, optional
        FTP connection (with logged in user "anonymous").

    Returns
    -------
    int
        The version number of the latest release.
    """
    if ftp is not None:
        assert isinstance(ftp, ftplib.FTP)

    close_connection = False
    if ftp is None:
        ftp_server = 'ftp.ensembl.org'
        user = 'anonymous'
        ftp = ftplib.FTP(ftp_server)
        ftp.login(user)
        close_connection = True

    data = []
    ftp.dir('pub', data.append)
    pat = re.compile(r'.* current_README -> release-(\d+)/README$')
    latest = []
    for d in data:
        m = pat.match(d)
        if m is not None:
            latest.append(int(m.group(1)))

    assert len(latest) == 1, len(latest)
    latest = latest[0]

    if close_connection:
        ftp.close()

    return latest


def _get_file_checksums(url: str, ftp: ftplib.FTP = None) -> Dict[str, str]:
    """Download and parse an Ensembl CHECKSUMS file and obtain checksums.

    Parameters
    ----------
    url : str
        The URL of the CHECKSUM file.
    ftp : `ftplib.FTP` or `None`, optional
        An FTP connection.
    
    Returns
    -------
    `collections.OrderedDict`
        An ordered dictionary containing file names as keys and checksums as
        values.

    Notes
    -----
    The checksums contains in Ensembl CHECKSUM files are obtained with the
    UNIX `sum` command.
    """
    # open FTP connection if necessary
    close_connection = False
    ftp_server = 'ftp.ensembl.org'
    ftp_user = 'anonymous'
    if ftp is None:
        ftp = ftplib.FTP(ftp_server)
        ftp.login(ftp_user)
        close_connection = True    
    
    # download and parse CHECKSUM file
    data = []
    ftp.retrbinary('RETR %s' % url, data.append)
    data = ''.join(d.decode('utf-8') for d in data).split('\n')[:-1]
    file_checksums = OrderedDict()
    for d in data:
        file_name = d[(d.rindex(' ') + 1):]
        sum_ = int(d[:d.index(' ')])
        file_checksums[file_name] = sum_
    
    _LOGGER.info('Obtained checksums for %d files', len(file_checksums))

    # close FTP connection if we opened it
    if close_connection:
        ftp.close()
    
    return file_checksums


def get_annotation_urls_and_checksums(species: str, release: int = None,
                                      ftp: ftplib.FTP = None):
    """Get FTP URLs and checksums for Ensembl genome annotations.
    
    Parameters
    ----------
    species : str or list of str
        The species or list of species for which to get genome annotations
        (e.g., "Homo_sapiens").
    release : int, optional
        The release number to look up. If `None`, use latest release. [None]
    ftp : ftplib.FTP, optional
        The FTP connection to use. If `None`, the function will open and close
        its own connection using user "anonymous".
    """
    ### open FTP connection if necessary
    close_connection = False
    ftp_server = 'ftp.ensembl.org'
    ftp_user = 'anonymous'
    if ftp is None:
        ftp = ftplib.FTP(ftp_server)
        ftp.login(ftp_user)
        close_connection = True    

    ### determine release if necessary
    if release is None:
        # use latest release
        release = get_latest_release(ftp=ftp)

    species_data = OrderedDict()
    if isinstance(species, str):
        species_list = [species]
    else:
        species_list = species
    for spec in species_list:

        # get the GTF file URL
        # => since the naming scheme isn't consistent across species,
        #    we're using a flexible scheme here to find the right file
        species_dir = '/pub/release-%d/gtf/%s' % (release, spec.lower())
        data = []
        ftp.dir(species_dir, data.append)
        gtf_file = []
        for d in data:
            i = d.rindex(' ')
            fn = d[(i + 1):]
            if fn.endswith('.%d.gtf.gz' % release):
                gtf_file.append(fn)
        assert len(gtf_file) == 1
        gtf_file = gtf_file[0]
        _LOGGER.debug('GTF file: %s', gtf_file)

        ### get the checksum for the GTF file
        checksum_url = '/'.join([species_dir, 'CHECKSUMS'])
        file_checksums = _get_file_checksums(checksum_url, ftp=ftp)
        gtf_checksum = file_checksums[gtf_file]
        _LOGGER.debug('GTF file checksum: %d', gtf_checksum)

        gtf_url = 'ftp://%s%s/%s' %(ftp_server, species_dir, gtf_file)

        species_data[spec] = (gtf_url, gtf_checksum)

    # close FTP connection, if we opened it
    if close_connection:
        ftp.close()

    return species_data
