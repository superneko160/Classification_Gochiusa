from icrawler.builtin import BingImageCrawler
import codeinfo

keywords_class = ["香風智乃", "保登心愛"]

for idx, cl in enumerate(codeinfo.CLASSES):
    crawler = BingImageCrawler(storage={"root_dir":f"opt/data/{cl}"})
    crawler.crawl(keyword=f'{keywords_class[idx]}', max_num=100)