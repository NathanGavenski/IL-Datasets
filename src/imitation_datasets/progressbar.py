'''
Created on 29 Jun 2011
Find the source and docs at https://github.com/thomaspurchas/tarfile-Progress-Reporter
@author: Thomas Purchas
'''
import tarfile
import os

from tqdm import tqdm


class Progressbar:
    def __init__(self, total: int = 0, description: str = None):
        self.total = total
        self.complete = 0
        self.pbar = tqdm(total=self.total, desc=description)

    def step(self, complete: float) -> None:
        '''
        This is an example callback function. If you pass this as the
        progress callback then it will print a progress bar to stdout.
        '''
        if complete == 100.0:
            self.pbar.update(1)


class sudotarinfo(object):
    size = None


class TarFile(tarfile.TarFile):
    '''
    classdocs
    '''
    def __init__(
        self,
        name=None, mode="r", fileobj=None, format=None,
        tarinfo=None, dereference=None, ignore_zeros=None, encoding=None,
        errors=None, pax_headers=None, debug=None, errorlevel=None
    ):
        '''
        Open an (uncompressed) tar archive `name'. `mode' is either 'r' to
        read from an existing archive, 'a' to append data to an existing
        file or 'w' to create a new file overwriting an existing one. `mode'
        defaults to 'r'.
        If `fileobj' is given, it is used for reading or writing data. If it
        can be determined, `mode' is overridden by `fileobj's mode.
        `fileobj' is not closed, when TarFile is closed.
        '''

        self.__progresscallback = None

        tarfile.TarFile.__init__(self, name, mode, fileobj, format,
                                 tarinfo, dereference, ignore_zeros, encoding,
                                 errors, pax_headers, debug, errorlevel)

    def add(self, name, arcname=None, recursive=True, filter=None, progress=None):
        '''
        Add the file *name* to the archive. *name* may be any type of file (directory,
        fifo, symbolic link, etc.). If given, *arcname* specifies an alternative name
        for the file in the archive. Directories are added recursively by default. This
        can be avoided by setting *recursive* to :const:`False`. If *exclude* is given
        it must be a function that takes one filename argument and returns a boolean
        value. Depending on this value the respective file is either excluded
        (:const:`True`) or added (:const:`False`). If *filter* is specified it must
        be a function that takes a :class:`TarInfo` object argument and returns the
        changed :class:`TarInfo` object. If it instead returns :const:`None` the :class:`TarInfo`
        object will be excluded from the archive. See :ref:`tar-examples` for an
        example.
        
        *progress* will be called with a signal integer with a value between 0 and 100,
        which represents the percentage of the file that has been added.
        '''

        if progress is not None and self.__progresscallback is None:
            tarinfo = tarfile.TarFile.gettarinfo(self, name, arcname)
            if tarinfo.isdir() and recursive:
                n_files = len(os.listdir(name))
                progress = Progressbar(total=n_files, description="Compressing files")
                self.__progresscallback = progress.step

        return tarfile.TarFile.add(
            self, name=name, arcname=arcname,
            recursive=recursive, filter=filter,
        )

    def addfile(self, tarinfo, fileobj=None, progress=None):
        """
        Add the TarInfo object `tarinfo' to the archive. If `fileobj' is
        given, tarinfo.size bytes are read from it and added to the archive.
        You can create TarInfo objects using gettarinfo().
        On Windows platforms, `fileobj' should always be opened with mode
        'rb' to avoid irritation about the file size.
        """

        if progress is not None:
            progress(0)
            self.__progresscallback = progress

        if fileobj is not None:
            fileobj = filewrapper(fileobj, tarinfo, self.__progresscallback)

        result = tarfile.TarFile.addfile(self, tarinfo, fileobj)

        #self.__progresscallback = None

        return result

    def extractall(self, path=".", members=None, progress=None):

        self.__progresscallback = None

        if progress is not None:
            original = self.fileobj
            try:
                stats = os.fstat(self.fileobj.fileno())
                sudoinfo = sudotarinfo()
                sudoinfo.size = stats.st_size
                self.fileobj = filewrapper(self.fileobj, sudoinfo, progress)
                self.__progresscallback = None
            except:
                # This means that we have a stream or similar. So we will report
                # file-by-file progress instead of total progress
                self.fileobj = original
                self.__progresscallback = progress

        result = tarfile.TarFile.extractall(self, path, members)

        self.fileobj = original

        return result

    def extract(self, member, path="", progress=None):

        if progress is not None:
            progress(0)
            self.__progresscallback = progress

        result = tarfile.TarFile.extract(self, member, path)

        #self.__progresscallback = None

        return result

    def extractfile(self, member, progress=None):
        """
        Extract a member from the archive as a file object. `member' may be
        a filename or a TarInfo object. If `member' is a regular file, a
        file-like object is returned. If `member' is a link, a file-like
        object is constructed from the link's target. If `member' is none of
        the above, None is returned.
        The file-like object is read-only and provides the following
        methods: read(), readline(), readlines(), seek() and tell()
        """

        if progress is not None:
            progress(0)
            self.__progresscallback = progress

        fileobj = tarfile.TarFile.extractfile(self, member)

        if fileobj is not None:
            fileobj = filewrapper(fileobj, member, self.__progresscallback)

        return fileobj

class filewrapper(object):
    '''
    This is a wrapper for a file object. I uses the __getattr__ function to cheat.
    '''

    def __init__(self, fileobj, tarinfo, progress):
        self._fileobj = fileobj

        self._size = tarinfo.size

        if self._size <= 0 or self._size is None:
            # Invalid size, we will not bother with the progress
            progress = None

        if progress is not None:
            progress(0)
        self._progress = progress
        self._lastprogress = 0

        self._totalread = 0

    def _updateprogress(self, length):
        '''
        Call this on every read to update our progress through the file
        '''
        if self._progress is not None:
            self._totalread += length

            progress = (self._totalread * 100) / self._size

            if progress > self._lastprogress and progress <= 100:
                self._progress(progress)
                self._lastprogress = progress

    def read(self, size= -1):
        data = self._fileobj.read(size)

        self._updateprogress(len(data))

        return data

    def readline(self, size= -1):
        data = self._fileobj.readline(size)

        self._updateprogress(len(data))

        return data

    def __getattr__(self, name):
        return getattr(self._fileobj, name)

    def __del__(self):
        self._updateprogress(self._size - self._totalread)

open = TarFile.open
