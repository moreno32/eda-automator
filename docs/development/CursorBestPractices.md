# **Cursor AI Best Practices for EDA Automator**

This guide provides essential guidelines for creating effective rules and prompts in Cursor AI for the EDA Automator project.

---

## **1. Structure for Rule Files (.mdc)**

### YAML Header
```yaml
---
description: "Clear purpose of the rule set"
globs: ["*.py", "*.ipynb"]
alwaysApply: true
---
```

### Key Components
- **Description**: Brief explanation of what the rules accomplish
- **Globs**: Target file patterns (Python files, notebooks, etc.)
- **Apply Setting**: Whether rules apply automatically or on demand

### Organization
- Group rules by related themes
- Use clear headings and bullet points
- Keep individual rules concise (1-3 lines)
- Use directive keywords: "Must", "Should", "Recommended"

---

## **2. Writing Effective Prompts**

### Best Practices
- **Be specific**: Request exactly what you need
- **Provide context**: Include relevant background information
- **Set clear expectations**: Define the desired outcome
- **Include examples**: Show sample code or expected results

### Example Structure
```
[Context] I'm working on the visualization module for EDA Automator.
[Request] Improve the histogram function to use the COLOR_PALETTES system.
[Constraints] Must handle sampling for large datasets.
[Example] Here's how a similar function is implemented: {code snippet}
```

---

## **3. Integration with EDA Automator**

### Visualization Standards
- Reference the `COLOR_PALETTES` dictionary for consistent styling
- Implement sampling thresholds for large datasets
- Use stratified sampling for preserving distributions
- Add sampling notes to charts when data is sampled

### Module Awareness
- Respect the separation of concerns between modules
- Maintain consistent interfaces across the project
- Follow existing patterns when extending functionality

---

## **4. Example: Visualization Rules**

```yaml
---
description: "Visualization standards for EDA Automator"
globs: ["*.py"]
alwaysApply: true
---

# Color Management
- "Always use COLOR_PALETTES dictionary for visualization consistency"
- "Select appropriate palette types: categorical, sequential, or diverging"
- "Implement fallback mechanisms for invalid palette requests"

# Data Sampling
- "Apply sampling when data exceeds the sampling_threshold"
- "Use stratified sampling for categorical data to preserve distributions"
- "Add sampling notes to all visualizations using sampled data"

# Chart Formatting
- "Maintain consistent axis labels, titles, and figure dimensions"
- "Optimize text readability with proper rotation and spacing"
- "Document visualization parameters in function docstrings"
```

---

## **5. Practical Examples**

### Improving Visualizations
```
Enhance the histogram function in univariate.py to:
1. Use the centralized COLOR_PALETTES system
2. Apply automatic sampling for datasets > 10,000 rows
3. Add sampling annotations when data is reduced
4. Maintain consistent styling with other visualization functions
```

### Custom Configuration
```
Create an example that demonstrates custom visualization configuration:
- Set sampling_threshold to 5000 rows
- Apply corporate color palette to all charts
- Enable stratified sampling by 'department' column
- Add clear annotations for all sampling operations
```

---

## **6. Quick Reference: Module Purposes**

- **data_quality.py**: Missing values, duplicates, outliers, quality metrics
- **stats_analysis.py**: Descriptive statistics, distribution tests, correlations
- **univariate.py**: Single variable analysis and visualization
- **bivariate.py**: Relationship analysis between variable pairs
- **multivariate.py**: Complex relationships and interactions
- **visuals.py**: Styling, color palettes, sampling, standardized plotting
- **utils.py**: Common utilities, type detection, logging
- **report.py**: HTML/Markdown report generation

---

## **7. Best Practices Summary**

- **Simplicity**: Create clear, focused rules and prompts
- **Consistency**: Follow established patterns and standards
- **Modularity**: Respect the project's module boundaries
- **Documentation**: Include examples and clear expectations
- **Visual Standards**: Apply consistent styling and sampling
- **Performance**: Consider large dataset handling approaches

---

## **8. Quality Assurance**

### Testing Focus Areas
- Unit testing for individual components
- Integration testing between modules
- Visual consistency across chart types
- Performance with varied dataset sizes

### Documentation Standards
- Complete function docstrings (NumPy style)
- Usage examples for complex features
- Clear parameter descriptions
- Return value specifications

