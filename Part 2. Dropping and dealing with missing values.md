## Can we safely drop the missing values? If not, how will you deal with them?

It depends on the attribute whether we can drop or dealing with missing values.

For example, bathrooms and bedroom counts will have default values of zero if data is missing. But in some cases these zero values are not concidered missing data. For studio apartments, the living room is concidered the sleeping space, and will correctly hold a value of zero. For all listing however, there must always be a bathroom. With this information, we can infer that listings with 0 bedrooms but >= 1 bathroom is correct data, but listings with 0 bedrooms and bathrooms is incorrect, and data must be removed. 

Instances with missing description fields can be kept, because listings with no descrption could help determine interest. 

Instances with missing _id values values should be kept, as _ids wont influence interest.
