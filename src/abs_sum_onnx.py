from sumy.parsers.plaintext_N import PlaintextParser_N
from sumy.parsers.plaintext_dot import PlaintextParser_DOT
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer


  

# doc = """
# Lĩnh vực Kỹ thuật và Công nghệ tiếp tục có GS.TSKH Nguyễn Đình Đức, ĐHQGHN. Ông là một trong những nhà khoa học đầu ngành của Việt Nam trong lĩnh vực Cơ học và vật liệu composite. Ông đã công bố trên 300 công trình khoa học, trong đó có 200 bài trên các tạp chí quốc tế ISI có uy tín. Bốn năm liên tiếp 2019, 2020, 2021 và 2022 ông lọt vào top 100.000 nhà khoa học có ảnh hưởng nhất thế giới. GS. Nguyễn Đình Đức vào tốp 94 thế giới trong lĩnh vực Engineering năm 2022, tức tốp 100 thế giới.
# """   

class abs_Sum():
    def __init__(self, lang='vietnamese'):
        self.lang = lang
    def summary_n(self, doc, k):
        tk = Tokenizer(self.lang)
        parser = PlaintextParser_N.from_string(doc, tk)
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, k)
        summary = [str(sentence) for sentence in summary]
        return '\n\n'.join((summary))
    def summary_dot(self, doc, k):
        tk = Tokenizer(self.lang)
        parser = PlaintextParser_DOT.from_string(doc, tk)
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, k)
        summary = [str(sentence) for sentence in summary]
        return '\n\n'.join((summary))
    
# doc = """
# PGS.TS Lê Hoàng Sơn công bố hơn 180 công trình, bài báo trên các tạp chí nước ngoài trong danh mục ISI. Ông là gương mặt lọt vào top 10.000 nhà khoa học xuất sắc của thế giới trong 4 năm liên tiếp 2019, 2020, 2021, 2022, đồng thời được gắn huy hiệu "Rising Star" - ngôi sao khoa học đang lên xuất sắc trên thế giới năm 2022.\n\nLĩnh vực Kỹ thuật và Công nghệ tiếp tục có GS.TSKH Nguyễn Đình Đức, ĐHQGHN. Ông là một trong những nhà khoa học đầu ngành của Việt Nam trong lĩnh vực Cơ học và vật liệu composite. Ông đã công bố trên 300 công trình khoa học, trong đó có 200 bài trên các tạp chí quốc tế ISI có uy tín. Bốn năm liên tiếp 2019, 2020, 2021 và 2022 ông lọt vào top 100.000 nhà khoa học có ảnh hưởng nhất thế giới. GS. Nguyễn Đình Đức vào tốp 94 thế giới trong lĩnh vực Engineering năm 2022, tức tốp 100 thế giới.\n\nLĩnh vực Khoa học Môi trường có GS.TS Phạm Hùng Việt và PGS.TS Từ Bình Minh, đều từ Trường Đại học Khoa học Tự nhiên, ĐHQGHN. GS. Phạm Hùng Việt hiện là Giám đốc Phòng thí nghiệm trọng điểm Công nghệ phân tích phục vụ kiểm định môi trường và An toàn thực phẩm, Trưởng nhóm nghiên cứu mạnh. Ông có hơn 100 công trình, bài báo công bố, sở hữu nhiều bằng sáng chế.\n\nPGS.TS Từ Bình Minh là nhà khoa học trong lĩnh vực hóa học. Chỉ trong hai năm 2019, 2020, nhóm nghiên cứu của ông đã công bố trên 20 công trình đăng trên các tạp chí quốc tế thuộc danh mục ISI uy tín, nhiều tạp chí trong số đó thuộc TOP 5% theo lĩnh vực chuyên sâu. Năm 2022, PGS.TS Từ Bình Minh cũng vào top nhà khoa học có ảnh hưởng nhất thế giới.
# """   
# ming = abs_Sum(lang='vietnamese')
# print(ming.summary_n(doc, 3))