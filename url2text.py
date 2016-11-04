import urllib, bs4, urlparse
import re, cPickle
from collections import deque

def url2text(url, fileout, countlimit=1, visited=None, BFS=False):
    '''Crawl a URL (following links in DFS/BFS manner) and grab all text.

    url [in]: (string) url.
    fileout [in]: (string) output file name.
    countlimit [in]: (integer) max web pages to crawl.
    visited [inout]: a set of strings of visited URLs
    BFS [in]: a boolean specifying if BFS or DFS is to be used for searching
    Return: nothing
    '''

    f = open(fileout, 'a')
    # ftmp = open('tmp.txt', 'w')
    
    if visited is None:
        visited = set([url])
    else:
        visited.add(url)
    url_queue = deque([url])
    count = 0
    #-- search
    while url_queue and count<countlimit:
        count += 1
        if BFS:
            url = url_queue.popleft()
        else:
            url = url_queue.pop()
        #-- read whole web page into a string
        html = urllib.urlopen(url).read()
        #-- parse the web page to a bs4 object
        soup = bs4.BeautifulSoup(html, 'html.parser')
        # text = soup.get_text()
        tags = soup.find_all('p')
        for tag in tags:
            text = tag.get_text()
            print >>f, text.encode('utf-8')
        #-- get all <a></a> tags into a list
        tags = soup.find_all('a')
        #-- scan all links and add to queue
        for tag in tags:
            #-- get the href attribute of a tag, keeping only 
            #--   up to path
            link = tag.get('href')
            #--- in case link is a relative path
            link = urlparse.urljoin(url, link)
            tmp = urlparse.urlparse(link)
            link = urlparse.urlunparse((tmp[0],tmp[1],tmp[2],'','',''))
            #-- crawl only certain URLs
            if link.startswith('https://docs.python.org/2') \
               and link.endswith(('htm','html','/')) \
               and link not in visited:
                # print >>ftmp, link[25:]
                visited.add(link)
                url_queue.append(link)
    #-- if while-loop exits due to count exceeding countlimit, 
    #--   remove the remaining urls in queue from visited
    #--   (since they are actually not visited)
    visited -= set(url_queue)
    print count
        
if __name__=='__main__':
#    url = 'https://docs.python.org/2/library/calendar.html' #https://docs.python.org/2/'
    url = 'https://docs.python.org/2/'
    fileout = 'raw_text.txt'
    countlimit = 2000
    try:
        visited = cPickle.load(open('visited_urls.pickle'))
    except:
        visited = set()
    url2text(url, fileout, countlimit=countlimit, visited=visited)
    cPickle.dump(visited, open('visited_urls.pickle', 'w'))

