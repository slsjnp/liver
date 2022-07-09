from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet

stylesheet = getSampleStyleSheet()
normalStyle = stylesheet['Normal']
story = []
story.append(Paragraph("Hello, ReportLab", normalStyle))
doc = SimpleDocTemplate('hello.pdf')
doc.build(story)
import datetime
import datetime
import subprocess
import codecs
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4, landscape
import reportlab.pdfbase.ttfonts

reportlab.pdfbase.pdfmetrics.registerFont(reportlab.pdfbase.ttfonts.TTFont('song', './SimHei.ttf'))
import reportlab.lib.fonts


def disk1_report():
    p1 = subprocess.Popen("cat ./1.log ", shell=True, stdout=subprocess.PIPE)
    return p1.stdout.readlines()


def create_pdf(input, output="disk1.pdf"):
    now = datetime.datetime.today()
    date = now.strftime("%h %d %Y %H:%M:%S")
    c = canvas.Canvas(output, pagesize=A4)
    c.setFont('song', 10)
    textobject = c.beginText()
    textobject.setTextOrigin(1 * inch, 11 * inch)
    textobject.textLines('''Disk Capacity Report: %s ''' % date)
    for line in input:
        textobject.textLine(line.strip())
    c.drawText(textobject)
    c.showPage()
    c.save()
    report = disk1_report()
    create_pdf(report)
create_pdf('hello')
