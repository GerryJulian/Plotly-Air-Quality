import plotly.express as px

df = px.data.gapminder().query("continent=='Oceania'")
fig = px.line(df, x="year", y="lifeExp", color='country')
fig.show()

# Get the current filename and change extension to .html
filename = os.path.splitext(os.path.basename(__file__))[0] + ".html"

# Save path: /project/charts
output_dir = os.path.join(os.path.dirname(__file__), "../charts")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, filename)

# Save HTML
fig.write_html(output_path, include_plotlyjs="cdn")

print(f"✅ {filename} generated at {output_dir}!")