This Project Only Can Run Locally


How to run this?

1. Clone Project
2. Run Locally the project by clicking main.html
3. Navigate Around



Important to understand :
1. Storyboard panel is inside folder /storyboard
2. Creating Charts using plotyly is inside folder  /scratch
3. Make sure your .py file have this code:

"# Get the current filename and change extension to .html
filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"

# Save path: /project/charts
output_dir = os.path.join(os.path.dirname(__file__), "../charts")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, filename)

# Save HTML
fig.write_html(output_path, include_plotlyjs="cdn")

print(f"✅ {filename} generated at {output_dir}!")"

4. Run your charts (Open terminal > using bash > run this "pyhton3 (namefile).py" or "python (namefile).py" )

5. Your .py file will generate an HTML file under charts folder
6. Include your charts by simply using iFrame under your storyboard file

example:
<div class="mb-5">
    <h4 class="fw-semibold text-center mb-3">Growth Trend</h4>
    <div class="ratio ratio-16x9 shadow border rounded-4 overflow-hidden">
        <iframe src="charts/chart2.html" title="Chart 2"></iframe>
    </div>
</div>

