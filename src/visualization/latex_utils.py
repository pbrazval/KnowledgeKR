import re

def personal_stargazer(mytable, textfolder, file_name, label, caption):
    latex_output = (mytable.
                to_latex(index=False, header=True, label = label, caption = caption, decimal = "."))
    file_name = textfolder + file_name
    with open(file_name, 'w') as f:
        f.write(latex_output)
    return None

def save_table_dual(figfolder, table, filename, tabular = False, row_names = False):
    tex_content = table.to_latex(index = row_names, label = f"tab:{filename}", position = "H!")
    
    if tabular:
        tex_content = re.search(r'\\begin{tabular}.*?\\end{tabular}', tex_content, re.DOTALL).group()

    with open(figfolder + filename + ".tex", 'w') as tex_file:
        tex_file.write(tex_content)

    # Print as HTML as well:
    with open(figfolder + filename + ".html", 'w') as html_file:
        html_file.write(table.to_html())

def get_quantile_term(maxkk):
    if maxkk == 3:
        return "tercile"
    elif maxkk == 4:
        return "quartile"
    elif maxkk == 5:
        return "quintile"
    elif maxkk == 10:
        return "decile"
    elif maxkk == 100:
        return "percentile"
    else:
        return f"{maxkk}-quantile"