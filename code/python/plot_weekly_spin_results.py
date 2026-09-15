"""Create publication-ready replacement Fig. 4e from weekly spin-test results.

Outputs vector SVG/PDF and a 600-dpi PNG. The line shows the observed weekly
Pearson correlation. Filled markers denote BH-FDR-significant spin tests; the
pink vertical band denotes the weeks surviving max-|r| family-wise correction.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "generated" / "weekly_spin_test" / "weekly_spin_results.csv"
OUT = ROOT / "results" / "generated" / "fig4e_weekly_spin"
OUT.mkdir(parents=True, exist_ok=True)

# Wide panel matching the aspect ratio of the assembled Fig. 4 layout.
W_MM, H_MM = 112.0, 66.0
PT_PER_MM = 72.0 / 25.4
W, H = W_MM * PT_PER_MM, H_MM * PT_PER_MM
DPI = 600
SCALE = DPI / 72.0

FONT_REG = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")

# Coral-red palette matched to the red/pink accents used in the other Fig. 4 panels.
INK = "#20242A"
GRID = "#D7DCE2"
BLUE = "#E04455"
BAND = "#F7DDE2"
WHITE = "#FFFFFF"


class SVG:
    def __init__(self): self.parts = []
    def rect(self,x,y,w,h,fill,stroke=None,sw=1): self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')
    def line(self,x1,y1,x2,y2,color,sw=1,dash=None): self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{sw}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def poly(self,pts,color,sw=1.5): self.parts.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in pts)+f'" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"/>')
    def circle(self,x,y,r,fill,stroke,sw=1): self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def text(self,x,y,s,size=7,bold=False,anchor='start',rotate=None):
        tr=f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ''
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial" font-size="{size}" font-weight="{700 if bold else 400}" fill="{INK}" text-anchor="{anchor}"{tr}>{escape(str(s))}</text>')
    def save(self,path):
        body='\n'.join(self.parts); path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_MM}mm" height="{H_MM}mm" viewBox="0 0 {W:.2f} {H:.2f}">\n<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n',encoding='utf-8')


class PDF:
    def __init__(self,path): self.c=pdfcanvas.Canvas(str(path),pagesize=(W,H)); self.c.setLineCap(1); self.c.setLineJoin(1)
    @staticmethod
    def rgb(h): return tuple(int(h[i:i+2],16)/255 for i in (1,3,5))
    def rect(self,x,y,w,h,fill,stroke=None,sw=1):
        self.c.setFillColorRGB(*self.rgb(fill)); self.c.setStrokeColorRGB(*self.rgb(stroke or fill)); self.c.setLineWidth(sw); self.c.rect(x,H-y-h,w,h,fill=1,stroke=int(stroke is not None))
    def line(self,x1,y1,x2,y2,color,sw=1,dash=None):
        self.c.setStrokeColorRGB(*self.rgb(color)); self.c.setLineWidth(sw); self.c.setDash([float(v) for v in dash.split(',')] if dash else []); self.c.line(x1,H-y1,x2,H-y2); self.c.setDash([])
    def poly(self,pts,color,sw=1.5):
        p=self.c.beginPath(); p.moveTo(pts[0][0],H-pts[0][1]); [p.lineTo(x,H-y) for x,y in pts[1:]]; self.c.setStrokeColorRGB(*self.rgb(color)); self.c.setLineWidth(sw); self.c.drawPath(p,stroke=1,fill=0)
    def circle(self,x,y,r,fill,stroke,sw=1): self.c.setFillColorRGB(*self.rgb(fill)); self.c.setStrokeColorRGB(*self.rgb(stroke)); self.c.setLineWidth(sw); self.c.circle(x,H-y,r,fill=1,stroke=1)
    def text(self,x,y,s,size=7,bold=False,anchor='start',rotate=None):
        self.c.saveState(); self.c.setFillColorRGB(*self.rgb(INK)); self.c.setFont('Arial-Bold' if bold else 'Arial',size); self.c.translate(x,H-y); self.c.rotate(-(rotate or 0)); width=self.c.stringWidth(str(s),'Arial-Bold' if bold else 'Arial',size); dx={'start':0,'middle':-width/2,'end':-width}[anchor]; self.c.drawString(dx,0,str(s)); self.c.restoreState()
    def save(self): self.c.showPage(); self.c.save()


class PNG:
    def __init__(self): self.im=Image.new('RGB',(round(W*SCALE),round(H*SCALE)),WHITE); self.d=ImageDraw.Draw(self.im)
    @staticmethod
    def q(v): return round(v*SCALE)
    def rect(self,x,y,w,h,fill,stroke=None,sw=1): self.d.rectangle((self.q(x),self.q(y),self.q(x+w),self.q(y+h)),fill=fill,outline=stroke,width=max(1,self.q(sw)))
    def line(self,x1,y1,x2,y2,color,sw=1,dash=None):
        if not dash: self.d.line((self.q(x1),self.q(y1),self.q(x2),self.q(y2)),fill=color,width=max(1,self.q(sw)))
        else:
            length=((x2-x1)**2+(y2-y1)**2)**.5; pattern=[float(v) for v in dash.split(',')]; pos=0.; draw=True
            while pos<length:
                step=pattern[int(pos//pattern[0])%len(pattern)] if len(pattern)==1 else pattern[int(draw==False)]; end=min(pos+step,length); xa=x1+(x2-x1)*pos/length; ya=y1+(y2-y1)*pos/length; xb=x1+(x2-x1)*end/length; yb=y1+(y2-y1)*end/length
                if draw: self.d.line((self.q(xa),self.q(ya),self.q(xb),self.q(yb)),fill=color,width=max(1,self.q(sw)))
                draw=not draw; pos=end
    def poly(self,pts,color,sw=1.5): self.d.line([(self.q(x),self.q(y)) for x,y in pts],fill=color,width=max(1,self.q(sw)),joint='curve')
    def circle(self,x,y,r,fill,stroke,sw=1): self.d.ellipse((self.q(x-r),self.q(y-r),self.q(x+r),self.q(y+r)),fill=fill,outline=stroke,width=max(1,self.q(sw)))
    def text(self,x,y,s,size=7,bold=False,anchor='start',rotate=None):
        font=ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG),round(size*SCALE)); box=self.d.textbbox((0,0),str(s),font=font); tw,th=box[2]-box[0],box[3]-box[1]
        if rotate is not None:
            layer=Image.new('RGBA',(tw+20,th+20),(255,255,255,0)); ld=ImageDraw.Draw(layer); ld.text((10,10),str(s),font=font,fill=INK); layer=layer.rotate(-rotate,expand=True); self.im.paste(layer,(self.q(x)-layer.width//2,self.q(y)-layer.height//2),layer); return
        ax={'start':0,'middle':tw/2,'end':tw}[anchor]; self.d.text((self.q(x)-ax,self.q(y)-th),str(s),font=font,fill=INK)
    def save(self,path): self.im.save(path,dpi=(DPI,DPI),compress_level=4)


def draw(c):
    d=pd.read_csv(SOURCE); ga=d.GA.to_numpy(); rr=d.pearson_r.to_numpy(); fdr=d.spin_q_BH_16weeks.to_numpy()<.05; fwer=d.spin_p_maxT_FWER.to_numpy()<.05
    L,R,T,B=42,9,10,38; x0,x1=L,W-R; y0,y1=T,H-B; ymin,ymax=-.7,.1
    X=lambda v:x0+(v-23)/(38-23)*(x1-x0); Y=lambda v:y1-(v-ymin)/(ymax-ymin)*(y1-y0)
    # Consecutive max-T-significant weeks 25-30 shown as a conservative window.
    c.rect(X(24.5),y0,X(30.5)-X(24.5),y1-y0,BAND)
    for yt in np.arange(-.6,.11,.2): c.line(x0,Y(yt),x1,Y(yt),GRID,.55); c.text(x0-5,Y(yt)+2.3,f'{yt:.1f}',6.5,anchor='end')
    c.line(x0,Y(0),x1,Y(0),INK,.7,dash='2,2')
    for xt in (23,26,29,32,35,38): c.line(X(xt),y1,X(xt),y1+2.5,INK,.65); c.text(X(xt),y1+11,str(xt),6.5,anchor='middle')
    c.line(x0,y0,x0,y1,INK,.8); c.line(x0,y1,x1,y1,INK,.8)
    pts=[(X(g),Y(r)) for g,r in zip(ga,rr)]; c.poly(pts,INK,1.25)
    for x,y,sig in zip([p[0] for p in pts],[p[1] for p in pts],fdr): c.circle(x,y,2.35,BLUE if sig else WHITE,BLUE if sig else INK,.9)
    c.text((x0+x1)/2,y1+21,'Gestational age (weeks)',8,bold=True,anchor='middle')
    c.text(14,(y0+y1)/2,'Pearson r (FDrad vs. curvature)',8,bold=True,anchor='middle',rotate=-90)
    # Compact legend below the axis title.
    ly=y1+32; c.circle(x0,ly,2.1,BLUE,BLUE,.8); c.text(x0+5,ly+2.1,'Spin BH-FDR q < 0.05',6.1)
    bx=x0+68; c.rect(bx,ly-3,8,6,BAND); c.text(bx+11,ly+2.1,'max-|r| FWER P < 0.05',6.1)


if not SOURCE.exists(): raise FileNotFoundError(SOURCE)
pdfmetrics.registerFont(TTFont('Arial',str(FONT_REG))); pdfmetrics.registerFont(TTFont('Arial-Bold',str(FONT_BOLD)))
svg=SVG(); draw(svg); svg.save(OUT/'Fig4e_weekly_FDrad_curvature_spin.svg')
pdf=PDF(OUT/'Fig4e_weekly_FDrad_curvature_spin.pdf'); draw(pdf); pdf.save()
png=PNG(); draw(png); png.save(OUT/'Fig4e_weekly_FDrad_curvature_spin_600dpi.png')
pd.read_csv(SOURCE).to_csv(OUT/'Fig4e_plot_data.csv',index=False)
print(OUT)
