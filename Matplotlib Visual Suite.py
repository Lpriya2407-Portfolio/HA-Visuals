"""
Healthcare Analytics: 25 Matplotlib Plots
Each plot has its own HA-based dataset and displays immediately with plt.show()
Includes comments on how to interpret the plot and its usefulness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
from scipy.stats import alpha

np.random.seed(42)

# ---------------- Helper ----------------
def currency(x, pos):
    if x >= 1e6: return f'{x/1e6:.1f}M'
    if x >= 1e3: return f'{x/1e3:.0f}k'
    return f'{x:.0f}'

# ====================== 1) Line Plot: Monthly Admissions ======================
months = pd.date_range('2024-01-01', periods=10, freq='MS')
trend = np.linspace(1000, 1600, len(months))
seasonal = 150 * np.sin(np.linspace(0, 3 * np.pi, len(months)))
noise = np.random.normal(0, 60, len(months))
admissions = trend + seasonal + noise
df1 = pd.DataFrame({'month': months, 'admissions': admissions})

plt.figure(figsize=(9,4))
plt.plot(df1['month'], df1['admissions'], marker='o', linewidth=2)
plt.title('Monthly Patient Admissions')
plt.xlabel('Month')
plt.ylabel('Admissions')
plt.grid(alpha=0.25)
plt.show()

# Interpretation & Use:
# - Shows trends over time with seasonal fluctuations
# - Useful to forecast staffing needs, resource allocation, and detect unusual spikes/dips

# ====================== 2) Scatter: Age vs Length-of-Stay ======================
n = 1000
ages = np.clip(np.random.normal(60,16,n),0,100)
los = np.clip(2 + 0.07*ages + np.random.normal(0,3,n),0.5,60)
readmit = np.random.binomial(1, p=0.08 + 0.001*ages, size=n)
df2 = pd.DataFrame({'age': ages, 'los': los, 'readmit': readmit})

plt.figure(figsize=(7,5))
for k, grp in df2.groupby('readmit'):
    plt.scatter(grp['age'], grp['los'], alpha=0.6, s=25, label=('No Readmit' if k==0 else 'Readmitted'))
plt.xlabel('Age')
plt.ylabel('Length of Stay (days)')
plt.title('Age vs Length-of-Stay')
plt.legend()
plt.grid(alpha=0.2)
plt.show()

# Interpretation & Use:
# - Scatter shows correlation between age and length-of-stay
# - Color indicates readmission risk
# - Useful for risk stratification, discharge planning, and identifying high-risk patients

# ====================== 3) Bar Plot: Top 10 Hospital Revenue ======================
hospitals = [f'Hospital_{i}' for i in range(1,25)]
revenue = np.random.gamma(5,2,len(hospitals))*1e6
df3 = pd.DataFrame({'hospital': hospitals, 'revenue': revenue})
top10 = df3.nlargest(10,'revenue').sort_values('revenue')

plt.figure(figsize=(8,5))
plt.barh(top10['hospital'], top10['revenue'],edgecolor='black',color='pink')
plt.title('Top 10 Hospitals by Revenue')
plt.xlabel('Revenue')
plt.gca().xaxis.set_major_formatter(FuncFormatter(currency))
plt.grid(axis='x', alpha=0.2)
plt.show()

# Interpretation & Use:
# - Horizontal bars show which hospitals generate most revenue
# - Helps management prioritize investments or analyze top-performing facilities

# ====================== 4) Grouped Bar: Department Admissions per Quarter ======================
departments = ['Emergency','Cardiology','Ortho','Oncology','Pediatrics']
quarters = ['Q1','Q2','Q3','Q4']
data = {q: np.random.randint(800,5000,len(departments)) for q in quarters}
df4 = pd.DataFrame(data, index=departments)

x = np.arange(len(departments))
width = 0.18
plt.figure(figsize=(13,5))
for i,q in enumerate(quarters):
    plt.bar(x + i*width, df4[q], width=width, label=q)
plt.xticks(x + width*1.5, departments, rotation=15)
plt.ylabel('Admissions')
plt.title('Quarterly Admissions by Department')
plt.legend()
plt.grid(axis='y', alpha=0.2)
plt.show()

# Interpretation & Use:
# - Each department has 4 bars showing quarterly admissions
# - Useful for seasonal workload planning and comparing departmental performance

# ====================== 5) Horizontal Bar: Patient Satisfaction ======================
hospitals5 = [f'Hosp_{i}' for i in range(1,11)]
satisfaction = np.clip(np.random.normal(3.9,0.4,len(hospitals5)),2.5,5)
df5 = pd.DataFrame({'hospital': hospitals5, 'satisfaction': satisfaction}).sort_values('satisfaction')

plt.figure(figsize=(8,5))
plt.barh(df5['hospital'], df5['satisfaction'])
plt.xlabel('Satisfaction Score (out of 5)')
plt.title('Patient Satisfaction by Hospital')
plt.xlim(2.5,5.0)
for i,v in enumerate(df5['satisfaction']):
    plt.text(v+0.02,i,f'{v:.2f}', va='center')
plt.show()

# Interpretation & Use:
# - Horizontal bars rank hospitals by patient satisfaction
# - Useful for quality improvement, identifying hospitals needing attention, or benchmarking

# ====================== 6) Stacked Bar: Insurance Mix by Hospital Category ======================
categories = ['Community','Regional','Specialty','Teaching']
ins_types = ['Private','Medicare','Medicaid','Uninsured']
data6 = {ins: np.random.randint(200,2000,len(categories)) for ins in ins_types}
df6 = pd.DataFrame(data6,index=categories)

bottom = np.zeros(len(df6))
plt.figure(figsize=(12,5))
for ins in ins_types:
    plt.bar(df6.index, df6[ins], bottom=bottom, label=ins,width=0.3)
    bottom += df6[ins].values
plt.title('Patient Insurance Mix by Hospital Category')
plt.ylabel('Patient Count')
plt.subplots_adjust(right=0.85)
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()

# Interpretation & Use:
# - Stacked bars show proportion of insurance types per hospital category
# - Useful to analyze payer mix and financial planning

# ====================== 7) Histogram: Lab Turnaround Time ======================
tat_minutes = np.clip(np.random.exponential(scale=25, size=1000),0.1,240)
df7 = pd.DataFrame({'tat_hours': tat_minutes/60})

plt.figure(figsize=(7,4))
plt.hist(df7['tat_hours'], bins=15, alpha=0.9,edgecolor='black')
plt.title('Lab Turnaround Time Distribution (hours)')
plt.xlabel('Hours')
plt.ylabel('Count')
plt.grid(alpha=0.15)
plt.show()

# # Interpretation & Use:
# # - Histogram shows distribution of lab TAT
# # - Useful to monitor efficiency and identify outliers
#
# # ====================== 8) Boxplot: Cost per Patient ======================
# departments8 = ['ER','Cardio','Ortho','Onco','Peds','Neurology']
# data8 = []
# for d in departments8:
#     base = {'ER':3000,'Cardio':12000,'Ortho':8000,'Onco':15000,'Peds':2500,'Neurology':11000}[d]
#     samples = np.random.lognormal(np.log(base),0.7,200)
#     data8.append(samples)
#
# plt.figure(figsize=(8,5))
# plt.boxplot(data8, labels=departments8, patch_artist=True)
# plt.title('Cost per Patient by Department')
# plt.ylabel('Cost (USD)')
# plt.gca().yaxis.set_major_formatter(FuncFormatter(currency))
# plt.show()
#
# # Interpretation & Use:
# # - Boxplots show median, quartiles, and outliers of patient cost per department
# # - Useful for budget planning and cost control
#
# # ====================== 9) Violin: Waiting Times by Triage ======================
# levels9 = ['Critical','Urgent','Semi-urgent','Non-urgent']
# data9 = []
# for lvl in levels9:
#     center = {'Critical':10,'Urgent':30,'Semi-urgent':60,'Non-urgent':120}[lvl]
#     data9.append(np.abs(np.random.normal(center, center*0.4, 500)))
#
# plt.figure(figsize=(8,5))
# plt.violinplot(data9, showmeans=True)
# plt.xticks([1,2,3,4], levels9)
# plt.ylabel('Waiting Time (min)')
# plt.title('Patient Waiting Time by Triage Level')
# plt.show()
#
# # Interpretation & Use:
# # - Violin plots show distribution and density of waiting times
# # - Useful to identify bottlenecks and improve patient flow
#
# # ====================== 10) Pie: Bed Occupancy by Department ======================
# labels10 = ['Medical','Surgical','ICU','Pediatrics','Maternity']
# values10 = np.random.randint(50,500,len(labels10)).astype(float)
# values10 /= values10.sum()
#
# plt.figure(figsize=(6,6))
# plt.pie(values10, labels=labels10, autopct='%1.1f%%', startangle=140,explode=[0,0.1,0,0,0],shadow=True)
# plt.axis('equal')
# plt.title('Bed Occupancy Share by Department')
# plt.show()
#
# # Interpretation & Use:
# # - Pie shows percentage of beds occupied per department
# # - Useful for capacity planning and resource allocation
#
# # ====================== 11) Stackplot: Patient Volume by Type ======================
# months11 = pd.date_range('2024-01-01', periods=18, freq='MS')
# types11 = ['Outpatient','Inpatient','ER']
# base11 = np.vstack([np.linspace(1200,1600,len(months11)),
#                     np.linspace(400,600,len(months11)),
#                     np.linspace(300,500,len(months11))])
# noise11 = np.random.normal(0,80, base11.shape)
# data11 = np.clip(base11 + noise11,0,None)
#
# plt.figure(figsize=(9,4))
# plt.stackplot(months11, data11, labels=types11,alpha=0.5)
# plt.legend(loc='upper left')
# plt.title('Patient Volume by Type')
# plt.ylabel('Patients')
# plt.show()
#
# # Interpretation & Use:
# # - Stackplot shows volume trend per patient type
# # - Useful for department load planning and trend analysis
#
# # ====================== 12) Step Plot: Cumulative Daily Admissions ======================
# days12 = np.arange(1,31)
# daily12 = np.random.poisson(80,len(days12))
# cumulative12 = np.cumsum(daily12)
#
# plt.figure(figsize=(8,4))
# plt.step(days12, cumulative12, where='pre', linewidth=2)
# plt.title('Cumulative Daily Admissions')
# plt.xlabel('Day')
# plt.ylabel('Cumulative Admissions')
# plt.grid(alpha=0.2)
# plt.show()
#
# # Interpretation & Use:
# # - Step plot shows cumulative admissions over days
# # - Useful to track progress and detect surges
#
# # ====================== 13) Stem Plot: Surgeries per Weekday ======================
# weekdays13 = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
# surgeries13 = np.random.randint(20,120,7)
#
# plt.figure(figsize=(7,4))
# plt.stem(weekdays13, surgeries13)
# plt.title('Surgeries per Weekday')
# plt.ylabel('Count')
# plt.grid(alpha=0.15)
# plt.show()
#
# # Interpretation & Use:
# # - Stem plot shows surgery counts by weekday
# # - Useful for weekly resource planning
#
# # ====================== 14) Hexbin: Patient Count vs Avg LOS ======================
# n14 = 2000
# patients14 = np.random.gamma(2,300,n14)
# los14 = np.random.normal(5+0.0008*patients14,1.8,n14)
#
# plt.figure(figsize=(7,5))
# plt.hexbin(patients14, los14, gridsize=40, cmap='Accent', mincnt=1,alpha=0.5)
# plt.xlabel('Patients')
# plt.ylabel('Avg LOS')
# plt.title('Hexbin: Patient Count vs Avg LOS')
# plt.colorbar(label='Counts')
# plt.show()
#
# # Interpretation & Use:
# # - Hexbin shows density of patient count vs LOS
# # - Useful to identify clusters and outlier hospitals
#
# # ====================== 15) Heatmap: KPI Correlation ======================
# kpis15 = ['ReadmitRate','Mortality','AvgLOS','Satisfaction','RevenuePerPatient','CostPerPatient']
# mat15 = np.random.normal(0,0.1,(6,6))
# corr15 = (mat15 + mat15.T)/2
# np.fill_diagonal(corr15,1)
#
# plt.figure(figsize=(10,8))
# plt.imshow(corr15,vmin=-1,vmax=1)
# plt.xticks(np.arange(6),kpis15, rotation=30, ha='right')
# plt.yticks(np.arange(6),kpis15)
# for i in range(6):
#     for j in range(6):
#         plt.text(j,i,f"{corr15[i,j]:.2f}",ha='center',va='center',fontsize=8)
# plt.title('Simulated KPI Correlation')
# plt.colorbar()
# plt.show()
#
# # Interpretation & Use:
# # - Heatmap shows correlation between KPIs
# # - Useful to identify related metrics and focus areas
#
# # ====================== 16) 3D Scatter: Age vs LOS vs Cost ======================
# n16 = 300
# age16 = np.clip(np.random.normal(60,15,n16),0,100)
# los16 = np.clip(np.random.normal(7,3,n16),1,30)
# cost16 = np.clip(los16*1000 + np.random.normal(0,2000,n16),500,50000)
# df16 = pd.DataFrame({'age':age16,'los':los16,'cost':cost16})
#
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111,projection='3d')
# sc=ax.scatter(df16['age'], df16['los'], df16['cost'], c=df16['cost'], cmap='viridis', s=40)
# ax.set_xlabel('Age')
# ax.set_ylabel('LOS (days)')
# ax.set_zlabel('Cost')
# ax.set_title('3D Scatter: Age vs LOS vs Cost')
# cbar = plt.colorbar(sc, pad=0.1)   # pad moves it slightly outside the plot
# cbar.set_label('Cost ($)')
# plt.show()
#
# # Interpretation & Use:
# # - 3D scatter shows relation between age, LOS, and cost
# # - Useful to find cost drivers and high-risk patients

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

np.random.seed(42)

# 17. Errorbar Plot: Revenue per Hospital with Std Dev
# n17 = 10
# revenue17 = np.random.randint(1000,5000,n17)
# hospitals17 = [f'Hospital_{i}' for i in range(1,n17+1)]
# std17 = np.random.randint(100, 500, n17)
# plt.figure(figsize=(8,5))
# plt.errorbar(hospitals17[:10], revenue17[:10], yerr=std17, fmt='o', color='teal', capsize=5)
# plt.xticks(rotation=45)
# plt.ylabel('Revenue ($)')
# plt.title('17: Revenue per Hospital with Std Dev')
# plt.show()
# Dot = hospital revenue, error bar = variability
# # HA Use: Identifies inconsistent revenue or anomalies
#
# # 18. Fill Between Plot: LOS Range per Month
# months18 = np.arange(1,13)
# low18 = np.clip(np.random.normal(5,1,12),1,10)
# high18 = np.clip(low18 + np.random.normal(3,1,12), low18+1,15)
# plt.figure(figsize=(8,5))
# plt.plot(months18, low18, label='Min LOS')
# plt.plot(months18, high18, label='Max LOS')
# plt.fill_between(months18, low18, high18, color='lightblue', alpha=0.4)
# plt.xlabel('Month')
# plt.ylabel('LOS (days)')
# plt.title('18: LOS Range Over Months')
# plt.legend()
# plt.show()
# # Filled area shows range of LOS
# # HA Use: Identifies months with unusually long stays → potential bottlenecks
#
# # 19.Mirrored Bar plot Style:Patients by Age and Gender

# age_groups = ['0-20','21-40','41-60','61-80','81+']
# male = np.random.randint(20,80,5)
# female = np.random.randint(20,80,5)
#
# y = np.arange(len(age_groups))
#
# plt.figure(figsize=(8,5))
# plt.barh(y, male, color='skyblue', label='Male')
# plt.barh(y, -female, color='pink', label='Female')  # negative for pyramid style
# xticks= plt.xticks()[0]  # get current ticks
# plt.xticks(xticks, [abs(int(x)) for x in xticks])  # convert to positive
# plt.yticks(y, age_groups)
# plt.xlabel('Number of Patients')
# plt.title('Patient Distribution by Age and Gender (Population Pyramid)')
# plt.legend()
# plt.grid(axis='x',linestyle='--',alpha=0.5)
#
# plt.show()

# # Interpretation:
# # - Left = female, Right = male
# # - Width = patient count per age group
# # HA Use:
# # - Visualize demographic distribution
# # - Identify age/gender trends for targeted care
#
# 20. Polar Plot: Avg Satisfaction per Month
# satisfaction20 = np.random.randint(1,6,12)
# angles20 = np.linspace(0, 2*np.pi, 12, endpoint=False)
# plt.figure(figsize=(6,6))
# plt.polar(angles20, satisfaction20, marker='o', color='purple')
# plt.title('20: Average Patient Satisfaction per Month')
# plt.show()
# Radial distance = satisfaction score
# HA Use: Reveals seasonal trends or dips in patient satisfaction
#
# 21. Contour Plot: Cost vs Age vs LOS
# x21 = np.linspace(0,100,30)
# y21 = np.linspace(1,30,30)
# x21, y21 = np.meshgrid(x21,y21)
# z21 = np.sin(x21/10)*np.cos(y21/5)*5000 + 10000
# plt.figure(figsize=(8,5))
# cp21 = plt.contourf(x21, y21, z21, cmap='viridis',levels=10)
# plt.colorbar(cp21, label='Cost ($)')
# plt.xlabel('Age')
# plt.ylabel('LOS (days)')
# plt.title('21: Contour: Cost by Age and LOS')
# plt.show()
# # Color = cost for each Age-LOS combination
# # HA Use: Identifies cost patterns across patients
#
# # 22. Quiver Plot

# np.random.seed(42)
#
# # Simulated HA data: 10 hospitals
# hospitals = [f'Hospital_{i}' for i in range(1,11)]
# revenue = np.random.randint(1000, 5000, 10)   # Revenue in $
# patients = np.random.randint(50, 200, 10)     # Number of patients
#
# # Simulate change vectors (delta revenue, delta patients)
# delta_revenue = np.random.randint(-500, 500, 10)  # change in revenue
# delta_patients = np.random.randint(-20, 20, 10)   # change in patients
# magnitude = np.sqrt(delta_revenue**2 + delta_patients**2)
#
# plt.figure(figsize=(8,6))
# plt.quiver(revenue, patients, delta_revenue, delta_patients, magnitude,
#            angles='xy', scale_units='xy', scale=1, cmap='viridis')
# plt.colorbar(label='Magnitude of Change')
# plt.xlabel('Revenue ($)')
# plt.ylabel('Number of Patients')
# plt.title('Patient Count vs Revenue Change Vector Field')
#
# # Optional: label each hospital
# for i, hosp in enumerate(hospitals):
#     plt.text(revenue[i]+4, patients[i]+4, hosp, fontsize=8,fontweight='bold')
#
# plt.xlim(revenue.min()-600, revenue.max()+600)
# plt.ylim(patients.min()-30, patients.max()+30)
# plt.grid(True,linestyle='--', alpha=0.5)
# plt.show()
#
# # 23. 3D Surface Plot: Cost vs Patients vs LOS
# x23 = np.linspace(50,500,30)
# y23 = np.linspace(1,30,30)
# x23, y23 = np.meshgrid(x23,y23)
# z23 = np.sin(x23/50)*np.cos(y23/5)*5000 + 15000
# fig23 = plt.figure(figsize=(8,6))
# ax23 = fig23.add_subplot(111, projection='3d')
# surf23 = ax23.plot_surface(x23, y23, z23, cmap=cm.viridis, edgecolor='none')
# ax23.set_xlabel('Patients')
# ax23.set_ylabel('LOS (days)')
# ax23.set_zlabel('Cost ($)')
# ax23.set_title('23: 3D Surface: Patients vs LOS vs Cost')
# cbar23 = plt.colorbar(surf23, pad=0.1)
# cbar23.set_label('Cost ($)')
# plt.show()
# # 3D surface shows how cost varies with patients and LOS
# # HA Use: Highlights regions of high cost & variable interactions
#
# # 24. Horizontal Stacked Bar: Beds Allocation per Hospital
# bed_types24 = np.random.randint(1,50,(10,3))  # ICU, General, Emergency
# plt.figure(figsize=(8,5))
# plt.barh(np.arange(10), bed_types24[:,0], color='red', label='ICU',alpha=0.5)
# plt.barh(np.arange(10), bed_types24[:,1], left=bed_types24[:,0], color='orange', label='General',alpha=0.5)
# plt.barh(np.arange(10), bed_types24[:,2], left=bed_types24[:,0]+bed_types24[:,1], color='green', label='Emergency',alpha=0.5)
# plt.yticks(np.arange(10), [f'Hospital_{i}' for i in range(1,11)])
# plt.xlabel('Number of Beds')
# plt.title('24: Beds Allocation per Hospital')
# plt.legend()
# plt.show()
# # Each bar = total beds; stacked segments = bed type
# # HA Use: Helps visualize bed allocation across hospitals

# # 25. Radial Bar Plot: Avg Monthly Patient Volume

# np.random.seed(42)
#
# months = np.arange(1,13)
# avg_patients = np.random.randint(50,150,12)
#
# theta = 2 * np.pi * (months-1)/12  # angles for 12 months
# radii = avg_patients
# width = 2*np.pi/12 * 0.8  # bar width
#
# plt.figure(figsize=(6,6))
# ax = plt.subplot(111, polar=True)
# bars = ax.bar(theta, radii, width=width, color='teal', alpha=0.7)
#
# ax.set_xticks(theta)
# ax.set_xticklabels([f'M{m}' for m in months])
# ax.set_yticklabels([])
# ax.set_title('Average Monthly Patient Volume (Radial Bar)')
#
# plt.show()
# Interpretation:
# - Bar length = patient volume
# - Angle = month
# HA Use:
# - Quickly see seasonal peaks in patient visits
# - Helps plan staffing and resource allocation

