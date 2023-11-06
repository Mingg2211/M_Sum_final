# coding: utf8

from transformers import BertTokenizer
import onnxruntime
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
import numpy as np

import sys
sys.path.append('.')
import os
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import sent_tokenize
import re

class M_Sum():
    def create_model_for_provider(self,folder_model_path: str, provider: str) -> InferenceSession: 

        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        model_path=''
        # Load the model as a graph and prepare the CPU backend 
        for file in os.listdir(folder_model_path):
            if file.endswith(".onnx"):
                model_path=os.path.join(folder_model_path,file)
            
        if model_path=='':
            return print("Could found model")
        session = InferenceSession(model_path, options, providers=[provider])
        session.disable_fallback()
            
        return session
    
    def __init__(self,lang='vi'):
        self.lang = lang
        self.pretrained = "model/ViBert/vi_Bert_onnx" if self.lang=='vi' \
        else("model/ChBert/ch_Bert_onnx" if self.lang=='ch' \
        else ("model/RuBert/ru_Bert_onnx" if self.lang=='ru' \
        else 'model/EnBert/en_Bert_onnx'))
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained, local_files_only=True)
        self.device = onnxruntime.get_device()
        if self.device == 'CPU':
            self.cpu_model = self.create_model_for_provider(self.pretrained, "CPUExecutionProvider")
        else : 
            self.cpu_model = self.create_model_for_provider(self.pretrained, "CUDAExecutionProvider")
            
    def sum_main(self, news, k):
        """
        news : dictionary bao gom :
            title : title cua news
            description : None(gan bang title) or string, description cua news
            paras :  doan tin cua news
        k : % van ban tom tat
        """
        title = news['title']
        description = news['description']
        paras = news['paras']
        paras = re.sub(r'\n+', '\n', paras)
        if self.lang == 'ch':
            if '\n' not in paras:
                paras = [s.replace('\u3000',' ').strip() for s in re.split('[。！？；]', paras)]
            else:
                paras = paras.split('\n')
        else :
            if '\n' not in paras:
                paras = sent_tokenize(paras)
                paras = [s.strip() for s in paras]
            else: 
                paras = paras.split('\n')
        print('Độ dài đoạn tin trước tóm tắt', len(paras))
        if len(paras)<6:
            return "\n\n".join(paras)
        else:
            if title.strip() != "" and description.strip() != "":
                input_id_title = self.tokenizer(title, return_tensors="pt",max_length=256, padding='max_length', truncation=True)
                inputs_title_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_title.items()}
                # Run the model (None = get all the return sents_vec_dict, centroid_docoutputs)
                _, title_pooled = self.cpu_model.run(None, inputs_title_onnx)
                
                # Inputs are provided through numpy array
                input_id_description = self.tokenizer(description, return_tensors="pt",max_length=256, padding='max_length', truncation=True)
                inputs_description_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_description.items()}
                # Run the model (None = get all the outputs)
                _, description_pooled = self.cpu_model.run(None, inputs_description_onnx)
                
                t_d = np.stack((title_pooled, description_pooled))
                centroid_doc = np.mean(t_d, axis=0)
                
                #vector sentences
                n = len(paras)
                sents_vec_dict = {v: k for v, k in enumerate(paras)}    
                for index in range(n) : 
                    input_id = self.tokenizer(paras[index], return_tensors="pt",max_length=256, padding='max_length', truncation=True)
                    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in input_id.items()}
                    # Run the model (None = get all the outputs)
                    _, pooled = self.cpu_model.run(None, inputs_onnx)
                    sents_vec_dict.update({index:pooled})
                    
                cosine_sim = {}
                for key in sents_vec_dict.keys():
                    cosine_2vec = cosine_similarity(centroid_doc, sents_vec_dict[key])
                    cosine_sim.update({key:cosine_2vec})
                final_sim = sorted(cosine_sim.items(), key=lambda x:x[1], reverse=True)
                print(final_sim)
                chossen = round(k*len(final_sim))
                list_index = dict(final_sim[:chossen]).keys()
                # print(list_index)
                result = []
                for index in sorted(list_index):
                    result.append(paras[index])
                print('Độ dài đoạn tin sau tóm tắt',len(result))
                return '\n\n'.join(result)
            elif title.strip() != "" and description.strip() == "":
                input_id_title = self.tokenizer(title, return_tensors="pt",max_length=256, padding='max_length', truncation=True)
                inputs_title_onnx = {k: v.cpu().detach().numpy() for k, v in input_id_title.items()}
                # Run the model (None = get all the return sents_vec_dict, centroid_docoutputs)
                _, title_pooled = self.cpu_model.run(None, inputs_title_onnx)
                centroid_doc = title_pooled
                #vector sentences
                n = len(paras)
                sents_vec_dict = {v: k for v, k in enumerate(paras)}    
                for index in range(n) : 
                    input_id = self.tokenizer(paras[index], return_tensors="pt",max_length=256, padding='max_length', truncation=True)
                    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in input_id.items()}
                    # Run the model (None = get all the outputs)
                    _, pooled = self.cpu_model.run(None, inputs_onnx)
                    sents_vec_dict.update({index:pooled})
                    
                cosine_sim = {}
                for key in sents_vec_dict.keys():
                    cosine_2vec = cosine_similarity(centroid_doc, sents_vec_dict[key])
                    cosine_sim.update({key:cosine_2vec})
                final_sim = sorted(cosine_sim.items(), key=lambda x:x[1], reverse=True)
                chossen = round(k*len(final_sim))
                list_index = dict(final_sim[:chossen]).keys()
                # print(list_index)
                result = []
                for index in sorted(list_index):
                    result.append(paras[index])
                print('Độ dài đoạn tin sau tóm tắt',len(result))
                return '\n\n'.join(result)
            elif title.strip() == "" and description.strip() == "":
                n = len(paras)
                sents_vec_dict = {v: k for v, k in enumerate(paras)}    
                for index in range(n) : 
                    input_id = self.tokenizer(paras[index], return_tensors="pt",max_length=256, padding='max_length', truncation=True)
                    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in input_id.items()}
                    # Run the model (None = get all the outputs)
                    _, pooled = self.cpu_model.run(None, inputs_onnx)
                    sents_vec_dict.update({index:pooled})
                X = list(sents_vec_dict.values())
                X = np.stack(X)
                centroid_doc = np.mean(X,0)  
                cosine_sim = {}
                for key in sents_vec_dict.keys():
                    cosine_2vec = cosine_similarity(centroid_doc, sents_vec_dict[key])
                    cosine_sim.update({key:cosine_2vec})
                final_sim = sorted(cosine_sim.items(), key=lambda x:x[1], reverse=True)
                chossen = round(k*len(final_sim))
                list_index = dict(final_sim[:chossen]).keys()
                # print(list_index)
                result = []
                for index in sorted(list_index):
                    result.append(paras[index])
            print('Độ dài đoạn tin sau tóm tắt',len(result))
            return '\n\n'.join(result)


if __name__ == '__main__':    
    ch_summ = M_Sum('ch')
    doc = """\n　　新华社北京10月31日电　题：不断夯实经济回升的基础——10月全国各地经济社会发展观察\n　　新华社记者\n　　投资项目加快落地、文体旅消费热度攀升、中小企业信心回升……随着一系列政策组合拳加快落地实施，10月份各地加快推进经济社会发展。进入四季度，各方坚定信心，落实好经济社会发展各项重点任务，不断夯实经济回升的基础，为实现全年发展目标打下坚实基础。\n\n　　观察之一：投资项目加速落地，大项目投资带动作用明显\n　　吊车高耸挥舞长臂，工人在井架构架上忙着拼接，施工车辆往来穿梭……10月份是入冬前最后的施工窗口期。为抢抓施工黄金期，鞍钢西鞍山铁矿项目正在加快施工现场的土地平整作业。\n　　“总投资229亿元的铁矿项目建成投产后，这里将成为绿色、智能、无废、无扰动的地下铁矿山。”西鞍山铁矿项目相关负责人说，通过“手递手”方式，项目矿权办理缩短了17个月，前期开工手续办理缩短了12个月。\n　　“十一”假期刚结束，多地接连召开会议，部署四季度工作，其中，加快推动投资项目落地是重点之一。\n　　放眼全国，一批重大项目正加速落地，发挥投资项目的带动作用：辽宁省目前入库储备项目超过1.7万个，总投资超8万亿元；陕西省四季度抓紧新开工600个以上省市重点项目；2023年四季度湖北省及武汉市重大项目10月26日集中开工，武汉市开工项目227个，总投资1861.7亿元……\n　　国家统计局固定资产投资统计司司长翟善清表示，前三季度，固定资产投资规模持续扩大，制造业投资增速继续加快。下阶段，要持续巩固投资增长态势，增强投资对优化供给结构的关键作用。\n\n　　【记者观察】进入四季度，各地正把项目建设作为稳住经济基本盘的重要抓手。继续保持抓高质量项目的定力和耐力，才能持续夯实稳的根基、蓄积新的动能。\n　　观察之二：文体旅消费加速，新供给激发新活力\n　　10月24日，南京夫子庙景区的科举博物馆广场前观者如潮。2023年全国射箭锦标赛（室外）在这里进行决赛，引来很多游客驻足观看。\n　　南京市秦淮区文化和旅游局（体育局）局长姜勇美说，让赛事活动与旅游文化场景深度融合，激发体育消费新潜力和新活力，能形成非常强的引流效应，进一步带动文旅消费。\n　　10月是传统的消费旺季。今年中秋国庆期间，全国超8亿人次出游、国内旅游收入超7500亿元；文体旅融合趋势明显，上海迪士尼乐园、西安秦始皇帝陵博物院、武汉黄鹤楼等文化地标成为热门目的地。\n　　10条体育旅游精品线路、9000万元商超百货消费券——为迎接第一届全国学生（青年）运动会到来，广西推出系列文体旅商促消费活动；首届湖北省户外运动大会开幕式在恩施举办，打造高山系列赛事，同时推出优质旅游线路和目的地……各地因地制宜推动商旅文体健相互赋能，开发新业态、新供给，促进新型消费潜力加速释放。\n\n　　【记者观察】进入四季度，各地抢抓黄金周机遇，通过供需双向发力推动消费升级，激发文体旅等消费新活力，带动消费尤其是服务消费持续回暖。随着政策效应持续释放，市场供给能力不断增强，消费引擎必将更加强劲。\n　　观察之三：秋收已近尾声，秋冬种抓紧推进\n　　金秋时节，在黑龙江省嫩江市伊拉哈镇，连片种植的大豆长势喜人，大型收割机在田间来回穿梭，一个来回就能装满七八吨大豆。\n　　“去年秋整地时政府给了一系列补贴政策，今年春播时又提供了根瘤菌促进大豆生长。我这13000亩大豆长势很好，预计亩产能到400斤以上。”嫩江市伊拉哈镇幸福乡村家庭农场理事长王宇心里挺踏实。\n　　在河北省磁县讲武城镇，小麦播种正抓紧进行。伴随着机械轰鸣声，立体匀播机一次性完成旋耕、播种，将小麦种在希望的田野。\n　　磁县农业农村局副局长杜文纲说，得益于北斗自动导航驾驶系统，匀播机可进行精准播种，提高出苗整齐度，为增产打下基础。\n　　当前，各地农业农村、发展改革等部门积极抓好秋收和秋冬种重点工作。安徽等地全力保障秋粮收储，备足仓容资金，确保“有仓收粮、有钱收粮、有人收粮、有车运粮”；江苏等地着力提升机械化播种质量，组建技术力量下沉一线加强技术和服务指导；黑龙江秋收已结束，正着手秸秆还田和综合利用，提出全省秸秆综合利用率达95%以上……\n　　农业农村部最新农情调度显示，目前秋收已进入扫尾阶段，秋粮大头已丰收到手，全年粮食产量有望再创历史新高。\n\n　　【记者观察】智慧高效、忙碌有序，这是记者在秋收秋种现场采访的最大感受。随着智慧农机和先进农业技术的推广应用，农业绿色发展、高质量发展的理念在广袤田野间加快落实。政策支持、科技支撑，人们辛勤耕耘，粮食供给保障能力稳步提升，丰收增产的基础不断夯实。\n　　观察之四：新能源车市场占有率提高，需求稳步释放\n　　一台台机械臂在数字化组装线上不断舞动，依次完成冲压、焊接、总装等环节……在重庆市赛力斯凤凰智慧工厂里，平均每两分钟就有一辆新能源车生产下线。\n　　“我们有一款新车型，发布1个多月以来订单量已达7万余台，工厂正开足马力保证订单交付。”赛力斯汽车相关负责人表示。\n　　“新能源车企加大产品创新力度，消费者认可度持续提升，市场占有率持续提高。”中国汽车工业协会副秘书长陈士华说。\n　　银川市启动2023金秋5000万元惠民消费券发放活动，其中发放购车消费补贴1000万元；武汉市10月13日发放1000万元“新能源汽车消费券”……当前，新能源汽车行业正值销售旺季，地方政府制定出台发放消费券、购车补贴等一系列措施。与此同时，各大车企也不断推出新产品，汽车市场需求稳步释放。\n\n　　【记者观察】新能源汽车的发展，关键在于便捷充电，未来应持续完善公共充电设施等，着力缓解消费者“充电难、充电慢”焦虑，筑牢新能源汽车消费基础。同时还应进一步瞄准基层市场，加快推动新能源汽车下乡。\n　　观察之五：稳增长措施落地见效，中小企业信心回升\n　　“这笔资金利率低、放款快，缓解了企业资金周转压力，增强了后续发展信心。”山东博凡教育科技有限公司董事长张庆磊说，不久前收到的300万元“政府采购合同融资贷款”放款，让企业购置新一批数字化设备的资金有了着落。\n　　“我们积极为政府采购供应商和金融机构搭建沟通桥梁，帮助中小微企业缓解资金压力大、融资难融资贵等问题。”山东省济宁市任城区政府采购监管工作分管负责人孟祥真说。\n　　随着一系列稳增长措施落地生效，中小企业发展信心有所回升。中国中小企业协会数据显示，三季度中小企业发展指数为89.2，比上季度上升0.2点，高于2022年同期水平。\n　　江西举办优质中小企业产融对接会，为中小企业和金融机构搭建桥梁；重庆区域性股权市场“专精特新”专板开板，首批入板211家企业；安徽省委、省政府在合肥市召开民营企业家恳谈会……10月份，各地出台一系列政策举措持续为中小企业发展注入新动力。\n\n　　【记者观察】前三季度，中小企业运行多项关键指标回升向好。近期，我国又对多项阶段性政策作出后续安排、着力激发民间投资活力、加快解决拖欠企业账款问题……一系列举措有利于稳定广大中小企业预期，增强发展信心，进一步激发企业创新活力，夯实稳增长的微观基础。\n　　统筹：杜宇、邹伟、邱红杰、王宇、刘心惠\n\n　　执笔记者：雷敏、严赋憬、魏玉坤\n\n　　采写记者：王聿昊、高亢、王雨萧、黄兴、赵鸿宇、王君宝、白涌泉、王恒志、张昕怡\n\n　　编辑|设计：刘羊旸、张虹生、潘一景\n\n　　视频编辑：杨牧\n"""
    news = {'title':'a',
            'description':'b',
            'paras':doc}
    s1 = ch_summ.sum_main(news,0.4)
    print(s1)
    print('----------------------------------------------------------------')
    ###############################
#     en_summ = M_Sum(lang='en')
#     doc = """
#     Yevgeny Prigozhin, the combative boss of Russia’s Wagner private military group, relishes his role as an anti-establishment maverick, but signs are growing that the Moscow establishment now has him pinned down and gasping for breath.

# Prigozhin placed a bet on his mercenaries raising the Russian flag in the eastern Ukrainian city of Bakhmut, albeit at a considerable cost to the ranks of his force and probably to his own fortune.

# He spent heavily on recruiting as many as 40,000 prisoners to throw into the fight, but after months of grinding battle and staggering losses he is struggling to replenish Wagner’s ranks, all the while accusing Russia’s Ministry of Defense of trying to strangle his force.

# Many analysts think his suspicions are well-founded – that Russia’s military establishment is using the Bakhmut “meat-grinder” to cut him down to size or eliminate him as a political force altogether.

# At the weekend, Prigozhin acknowledged that the battle in Bakhmut was “difficult, very difficult, with the enemy fighting for each meter.”

# In another video message, Prigozhin said: “We need the military to shield the approaches (to Bakhmut). If they manage to do so, everything will be okay. If not, then Wagner will be encircled together with the Ukrainians inside Bakhmut.”
#     """
#     news = {'title':'',
#             'description':'',
#             'paras':doc}
#     s2 = en_summ.sum_main(news,0.4)
#     print(s2)
#     print('----------------------------------------------------------------')
    
#     ##################################
#     ru_summ = M_Sum(lang='ru')
#     doc = """
#     В суде Лерчек и ее супруг просили суд не лишать из интернета, поскольку на нем завязан весь их бизнес - помимо самого блога это еще и фирма по производству косметики. Чекалин объявил, что у них официально трудоустроены 250 человек.

# Но судья согласилась с доводами следствия: с помощью интернета, мобильной и телефонной связи супруги могут оказывать влияние на свидетелей.

# Валерия уехала домой сразу после заседания, а Артему (решение по нему выносилось отдельно) пришлось задержаться.
#     """
#     news = {'title':'',
#             'description':'',
#             'paras':doc}
#     s3 = ru_summ.sum_main(news,0.4)
#     print(s3)
#     print('----------------------------------------------------------------')
    
#     #########################################
#     ch_summ = M_Sum(lang='ch')
#     doc = """
#     尊敬的各位同事，女士们，先生们，朋友们：欢迎大家来到西安，出席中国－中亚峰会，共商中国同中亚五国合作大计。西安古称长安，是中华文明和中华民族的重要发祥地之一，也是古丝绸之路的东方起点。2100多年前，中国汉代使者张骞自长安出发，出使西域，打开了中国同中亚友好交往的大门。千百年来，中国同中亚各族人民一道推动了丝绸之路的兴起和繁荣，为世界文明交流交融、丰富发展作出了历史性贡献。中国唐代诗人李白曾有过“长安复携手，再顾重千金”的诗句。今天我们在西安相聚，续写千年友谊，开辟崭新未来，具有十分重要的意义。2013年，我担任中国国家主席后首次出访中亚，提出共建“丝绸之路经济带”倡议。10年来，中国同中亚国家携手推动丝绸之路全面复兴，倾力打造面向未来的深度合作，将双方关系带入一个崭新时代。横跨天山的中吉乌公路，征服帕米尔高原的中塔公路，穿越茫茫大漠的中哈原油管道、中国－中亚天然气管道，就是当代的“丝路”；日夜兼程的中欧班列，不绝于途的货运汽车，往来不歇的空中航班，就是当代的“驼队”；寻觅商机的企业家，抗击新冠疫情的医护人员，传递友谊之声的文化工作者，上下求索的留学生，就是当代的友好使者。中国同中亚国家关系有着深厚的历史渊源、广泛的现实需求、坚实的民意基础，在新时代焕发出勃勃生机和旺盛活力。各位同事！当前，百年变局加速演进，世界之变、时代之变、历史之变正以前所未有的方式展开。中亚是亚欧大陆的中心，处在联通东西、贯穿南北的十字路口。世界需要一个稳定的中亚。中亚国家主权、安全、独立、领土完整必须得到维护，中亚人民自主选择的发展道路必须得到尊重，中亚地区致力于和平、和睦、安宁的努力必须得到支持。世界需要一个繁荣的中亚。一个充满活力、蒸蒸日上的中亚，将实现地区各国人民对美好生活的向往，也将为世界经济复苏发展注入强劲动力。世界需要一个和谐的中亚。“兄弟情谊胜过一切财富”。民族冲突、宗教纷争、文化隔阂不是中亚的主调，团结、包容、和睦才是中亚人民的追求。任何人都无权在中亚制造不和、对立，更不应该从中谋取政治私利。世界需要一个联通的中亚。中亚拥有得天独厚的地理优势，有基础、有条件、有能力成为亚欧大陆重要的互联互通枢纽，为世界商品交换、文明交流、科技发展作出中亚贡献。各位同事！去年，我们举行庆祝中国同中亚五国建交30周年视频峰会时，共同宣布建设中国－中亚命运共同体。这是我们在新的时代背景下，着眼各国人民根本利益和光明未来，作出的历史性选择。建设中国－中亚命运共同体，要做到四个坚持。一是坚持守望相助。我们要深化战略互信，在涉及主权、独立、民族尊严、长远发展等核心利益问题上，始终给予彼此明确、有力支持，携手建设一个守望相助、团结互信的共同体。二是坚持共同发展。我们要继续在共建“一带一路”合作方面走在前列，推动落实全球发展倡议，充分释放经贸、产能、能源、交通等传统合作潜力，打造金融、农业、减贫、绿色低碳、医疗卫生、数字创新等新增长点，携手建设一个合作共赢、相互成就的共同体。三是坚持普遍安全。我们要共同践行全球安全倡议，坚决反对外部势力干涉地区国家内政、策动“颜色革命”，保持对“三股势力”零容忍，着力破解地区安全困境，携手建设一个远离冲突、永沐和平的共同体。四是坚持世代友好。我们要践行全球文明倡议，赓续传统友谊，密切人员往来，加强治国理政经验交流，深化文明互鉴，增进相互理解，筑牢中国同中亚国家人民世代友好的基石，携手建设一个相知相亲、同心同德的共同体。各位同事！这次峰会为中国同中亚合作搭建了新平台，开辟了新前景。中方愿以举办这次峰会为契机，同各方密切配合，将中国－中亚合作规划好、建设好、发展好。一是加强机制建设。我们已经成立外交、经贸、海关等会晤机制和实业家委员会。中方还倡议成立产业与投资、农业、交通、应急管理、教育、政党等领域会晤和对话机制，为各国开展全方位互利合作搭建广泛平台。二是拓展经贸关系。中方将出台更多贸易便利化举措，升级双边投资协定，实现双方边境口岸农副产品快速通关“绿色通道”全覆盖，举办“聚合中亚云品”主题活动，打造大宗商品交易中心，推动贸易规模迈上新台阶。三是深化互联互通。中方将全面提升跨境运输过货量，支持跨里海国际运输走廊建设，提升中吉乌、中塔乌公路通行能力，推进中吉乌铁路项目对接磋商。加快现有口岸现代化改造，增开别迭里口岸，大力推进航空运输市场开放，发展地区物流网络。加强中欧班列集结中心建设，鼓励优势企业在中亚国家建设海外仓，构建综合数字服务平台。四是扩大能源合作。中方倡议建立中国－中亚能源发展伙伴关系，加快推进中国－中亚天然气管道D线建设，扩大双方油气贸易规模，发展能源全产业链合作，加强新能源与和平利用核能合作。五是推进绿色创新。中方愿同中亚国家在盐碱地治理开发、节水灌溉等领域开展合作，共同建设旱区农业联合实验室，推动解决咸海生态危机，支持在中亚建立高技术企业、信息技术产业园。中方欢迎中亚国家参与可持续发展技术、创新创业、空间信息科技等“一带一路”专项合作计划。六是提升发展能力。中方将制定中国同中亚国家科技减贫专项合作计划，实施“中国－中亚技术技能提升计划”，在中亚国家设立更多鲁班工坊，鼓励在中亚的中资企业为当地提供更多就业机会。为助力中国同中亚国家合作和中亚国家自身发展，中方将向中亚国家提供总额260亿元人民币的融资支持和无偿援助。七是加强文明对话。中方邀请中亚国家参与“文化丝路”计划，将在中亚设立更多传统医学中心，加快互设文化中心，继续向中亚国家提供政府奖学金名额，支持中亚国家高校加入“丝绸之路大学联盟”，办好中国同中亚国家人民文化艺术年和中国－中亚媒体高端对话交流活动，推动开展“中国－中亚文化和旅游之都”评选活动、开行面向中亚的人文旅游专列。八是维护地区和平。中方愿帮助中亚国家加强执法安全和防务能力建设，支持各国自主维护地区安全和反恐努力，开展网络安全合作。继续发挥阿富汗邻国协调机制作用，共同推动阿富汗和平重建。各位同事！去年十月，中国共产党第二十次全国代表大会成功召开，明确了全面建成社会主义现代化强国、实现第二个百年奋斗目标、以中国式现代化全面推进中华民族伟大复兴的中心任务，绘就了中国未来发展的宏伟蓝图。我们愿同中亚国家加强现代化理念和实践交流，推进发展战略对接，为合作创造更多机遇，协力推动六国现代化进程。各位同事！中国陕西有句农谚，“只要功夫深，土里出黄金”。中亚谚语也说，“付出就有回报，播种就能收获”。让我们携手并肩，团结奋斗，积极推进共同发展、共同富裕、共同繁荣，共同迎接六国更加美好的明天！谢谢大家。
#     """
#     news = {'title':'习近平在中国－中亚峰会上的主旨讲话（全文）',
#             'description':'',
#             'paras':doc}
#     s4 = ch_summ.sum_main(news,0.4)
#     print(s4)