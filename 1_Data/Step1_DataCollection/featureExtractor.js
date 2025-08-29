// #########################################################

// Statistics:

// Total Pages: 1590
// Total Pages that has tables saved: 1406
// Total Tables Count: 7304
// Total Tables Saved: 4186
// Total Tables with Error: 1397
// Total Tables hidden: 0
// Total Tables with less than two rows: 1721

// Program Ends
// *******************************

const puppeteer = require('puppeteer')
var fs = require('fs')
const XLSX = require('xlsx'); // Import the xlsx library

// Create a new workbook and worksheet
var workbook = XLSX.utils.book_new();
var worksheet = XLSX.utils.aoa_to_sheet([['page_number', 'page_name', 'table_no', 'page_link', 'rows', 'cols',
'th','height','width','y','x', 'is_leaf_table','total_numbers','total_textContent','has_form_elements',
'has_row_span','has_col_span', 'last_row_background','first_row_background','table_background','img_tag_in_table',
'table_border_collapse' ,'table_border', 'border_width', 'border_style', 'border_color', 'last_col_width',
'last_col_height', 'last_row_height','last_row_width','first_col_height','first_row_height','first_col_width',
'first_row_width', 'cols_in_last_row', 'cols_in_first_row']]); // Header row


const regex = /(?:https?:\/\/)?(?:www\.)?([a-zA-Z0-9.-]+)\./;


const links = readLinksFromFile("links1.txt");


    
(async () => {
        
        
    const browser = await puppeteer.launch();
        //const processLink = async (link, index) => {
    var index = 0; 
    var total_Pages = 0;
    var total_tables_count = 0;
    var total_tables_saved = 0;
    var total_tables_with_error = 0;
    var total_hidden_tables = 0;
    var total_tables_withLessThanTwoRows = 0;
    for await (const link of links) { 
        try{  
        const page = await browser.newPage();
        pageLink = link;
        const match = pageLink.match(regex);
        if (match && match[1]) 
        { 
                var name = match[1].replace(/^[^.]*\./, '');
        }
        pageName = (index+1) + '_' + name;;
        var fullPath = "webPages/"+pageName; // this is the folder where all the screenshots of tables will be saved 

        
            await page.goto(pageLink, { waitUntil: 'networkidle2' });
            await page.setViewport({ width: 1920, height: 1080 });
        

            new Promise(r => setTimeout(r, 5000));



            process.stdout.write("\n#########################################################\n");
            process.stdout.write("Starting Feature Extraction for page: " + pageName + "\n\n");
            

            //await page.screenshot({ path: "fullPage.png" });
            const elementHandles = await page.$$("*");
        
        var i = 0;
        var tableNumber = 1;
        
        // this is to increment the index when the tables in the page have been processed
        // sometimes, the folder is created, but it's empty and has no table, this flag
        // to avoid this problem. 
        var pageHasTableFlag = 0; 
        var folderAlreadyCreated = 0;
        while (i < elementHandles.length) {

            var tag_name = await (await elementHandles[i].getProperty('tagName')).jsonValue();
            
            if (tag_name.toLowerCase() === 'table') {

                total_tables_count ++; 


                var innerHtmlJSHandle = await elementHandles[i].getProperty('innerHTML');                
                var innerHTML = (await innerHtmlJSHandle.jsonValue()).replace(/\s+/g, ' ').trim();

                // Finding the row count 
                var rowCount = numberOfRows(innerHTML);
                
                //=================================================================================
            

                var minRows = 2;
                // Count how many pages In total 
                // Count how many pages visited
                // Count how many tables in total
                // Error occured with how many
                // Count how many has less than 2 rows 
                // count many are hidden 
                if (elementHandles[i].isVisible() && rowCount >= minRows) {
                    try {
                        // ----------------------------------------------------------------------//
                        process.stdout.write("\nPage Name: " + pageName + ', Table: ' + tableNumber);
                        process.stdout.write("\n*************************************\n\n"); 
                        
                        // ----------------------------------------------------------------------//
                        
                        
                        // ----------------------------------------------------------------------//
                        // Finding the col count, and the img tag count 
                        var colCount = await numberOfColumns(elementHandles[i]);
                        var numOfImgs = numberOfImages(innerHTML);
                        // process.stdout.write(' \n\nrowCount: ' + rowCount);
                        // process.stdout.write(' \ncolCount: ' + colCount);
                        // process.stdout.write(' \nnumOfImgs: ' + numOfImgs);
                        // ----------------------------------------------------------------------//
                        
                        // Finding the th tag count 
                        var th_tags = numberOfthTags(innerHTML);
                        // process.stdout.write('\nThe table has : ' + th_tags + ' th tags');
                        // ----------------------------------------------------------------------//
                        
                        // Finding how many form elements the table has 
                        var has_form_elements = hasFormElement(innerHTML);
                        // process.stdout.write('\nThe table has form elements: ' + has_form_elements);
                        // ----------------------------------------------------------------------//

                        // Checking if the table is leaf or no
                        var is_leaf_table = await isLeaf(elementHandles[i]);
                        // process.stdout.write('\nThe table is leaf: ' + is_leaf_table);
                        // ----------------------------------------------------------------------//

                        // Getting the x,y,height, and width
                        const rectObject =  await elementHandles[i].boundingBox();
                        var x = rectObject.x;
                        var y = rectObject.y;
                        var width = rectObject.width;
                        var height = rectObject.height;
                        // process.stdout.write(' x: ' + rectObject.x);
                        // process.stdout.write(' y: ' + rectObject.y);
                        // process.stdout.write('\nTable width: ' + rectObject.width);
                        // process.stdout.write('\nTable height: ' + rectObject.height);
                        
                        // ----------------------------------------------------------------------//

                        // Getting table background color
                        const tableBackgroundColor = await elementHandles[i].evaluate(table => {
                            const computedStyle = getComputedStyle(table);
                            return computedStyle.backgroundColor;
                        });
                        // process.stdout.write('\nThe table background is : '+ tableBackgroundColor);
                        // ----------------------------------------------------------------------//

                        // Border Collapse value 
                        const borderCollapseValue = await elementHandles[i].evaluate((table) => {
                            const computedStyle = getComputedStyle(table);
                            return computedStyle.getPropertyValue('border-collapse');
                        });
                        // process.stdout.write('\nBorder Collapse Value: '+ borderCollapseValue);
                        // ------------------------------------------------------------------//
                        
                        /// Border properties 
                        const tableBorderStyle = await elementHandles[i].evaluate((table) => {
                            const computedStyle = getComputedStyle(table);
                            return computedStyle.getPropertyValue('border');
                        });
                        // process.stdout.write('\ntableBorderStyle: '+ tableBorderStyle);  
                        
                        // ------------//
                        // Saving each property alone

                        const tableBorderProperties = await elementHandles[i].evaluate((table) => {
                        const computedStyle = getComputedStyle(table);

                        return {
                            borderWidth: computedStyle.borderLeftWidth,
                            borderColor: computedStyle.borderLeftColor,
                            borderStyle: computedStyle.borderLeftStyle,
                        };
                        });
                        var borderWidth = tableBorderProperties.borderWidth;
                        var borderColor = tableBorderProperties.borderColor;
                        var borderStyle = tableBorderProperties.borderStyle;

                        // process.stdout.write('\nborderWidth: '+ borderWidth); 
                        // process.stdout.write('\nborderColor: '+ borderColor); 
                        // process.stdout.write('\nborderStyle: '+ borderStyle); 

                        // ----------------------------------------------------------------------//

                        // Finding the count of row spans and col spans
                        const rowColSpans = await findSpans(elementHandles[i]);
                        const row_span = rowColSpans.rowspanCount;
                        const col_span = rowColSpans.colspanCount;
                        // process.stdout.write('\nRow Span: '+ row_span);
                        // process.stdout.write('\nCol Span: '+ col_span);
                        // ----------------------------------------------------------------------//

                        // Here I'm getting all the data written in the table cells  
                        const tableData = await getData(elementHandles[i]);
                        // process.stdout.write('style: '+tableData);
                        
                        // I use the data I got above to count the words 
                        var wordFrequency = wordsFrequency(tableData);
                        // process.stdout.write('\nwordsFrequency: ' + wordFrequency);
                        // -------------------

                        // I use the data I got above to count the numbers 
                        var numberFrequency = numbersFrequency(tableData);
                        // process.stdout.write('\nnumbersFrequency: ' + numberFrequency);
                        
                        // ----------------------------------------------------------------------//

                        // Getting the features of the first row
                        const first_row_features = await getFirstRow(elementHandles[i]);
                        const first_row_width = first_row_features.width;
                        const first_row_height = first_row_features.height;
                        const columns_in_first_row = first_row_features.columns;
                        const first_row_background = first_row_features.backgroundColor;
                        // process.stdout.write('\nfirst row width is : ' + first_row_width);
                        // process.stdout.write('\nfirst row height is : ' + first_row_height);
                        // process.stdout.write('\nColumns in first row is : ' + columns_in_first_row);    
                        // process.stdout.write('\nfirst row background is : ' + first_row_background);  
                        
                        // ----------------------------------------------------------------------//

                        // Getting the features of the last row
                        const last_row_features = await getLastRow(elementHandles[i]);
                        const last_row_width = last_row_features.width;
                        const last_row_height = last_row_features.height;
                        const columns_in_last_row = last_row_features.columns;
                        const last_row_background = last_row_features.backgroundColor;
                        // process.stdout.write('\nlast row width is : ' + last_row_width);
                        // process.stdout.write('\nlast row height is : ' + last_row_height);
                        // process.stdout.write('\nColumns in last row is : ' + columns_in_last_row);    
                        // process.stdout.write('\nlast row background is : ' + last_row_background);

                        // ----------------------------------------------------------------------//

                        // Getting the features of the first column
                        const first_col_features = await getFirstCol(elementHandles[i]);
                        const first_col_height = first_col_features.height;
                        const first_col_width = first_col_features.width;
                        // process.stdout.write('\nfirst col height: ' + first_col_height);
                        // process.stdout.write('\nfirst col width: ' + first_col_width);

                        // ----------------------------------------------------------------------//

                        // Getting the features of the last column 
                        const last_col_features = await getLastCol(elementHandles[i]);
                        const last_col_height = last_col_features.height;
                        const last_col_width = last_col_features.width;
                        // process.stdout.write('\nLast col height: ' + last_col_height);
                        // process.stdout.write('\nLast col width: ' + last_col_width);
                        
                        // ----------------------------------------------------------------------//
                        
                        // ***** SAVING THE FEATURES TO THE EXCEL SHEET *****// 

                        // Append the features to the worksheet
                        XLSX.utils.sheet_add_aoa(worksheet, [[index+1, pageName, tableNumber,  pageLink, rowCount, colCount, 
                        th_tags, height, width, y, x, is_leaf_table, numberFrequency, wordFrequency,
                        has_form_elements, row_span, col_span, last_row_background, first_row_background, 
                        tableBackgroundColor, numOfImgs, borderCollapseValue, tableBorderStyle, borderWidth, 
                        borderStyle, borderColor, last_col_width, last_col_height, last_row_height, 
                        last_row_width, first_col_height,first_row_height, first_col_width, first_row_width, 
                        columns_in_last_row, columns_in_first_row]], { origin: -1 });

                        // ----------------------------------------------------------------------//
                        
                        //process.stdout.write('\nTable ' + tableNumber + ' From page' + pageCount + ' has been added to the Excel sheet');
                        process.stdout.write("Features are saved");
                        process.stdout.write("\n-------------------------------\n");
                        // ----------------------------------------------------------------------//

                        // After the features are saved and no error occured, create a folder 
                        //Creating a folder for the page
                        if (folderAlreadyCreated != 1)
                        {
                            try {
                                fs.mkdirSync(fullPath);
                                folderAlreadyCreated = 1;
                            } catch (error) {
                                process.stdout.write(' #Exception: ' + error);
                            }
                        }
                        var pageHasTableFlag = 1;
                        
                        // ----------------------------------------------------------------------//
                        // take a screenshot of the table and save it in the folder
                        await elementHandles[i].screenshot({path: fullPath + '/' + 'table' + tableNumber.toString() + '.png'});
                        
                        process.stdout.write("The screenshot has been taken");
                        process.stdout.write("\n=====================================\n");
                        // ----------------------------------------------------------------------//
                        
                        // Increment the table count
                        tableNumber++;

                        total_tables_saved++; 
                        //---------------------------------------------------------------------------//
                    } catch (error) {
                        // It comes here in case any error occurs while getting the features.
                        total_tables_with_error++;
                        process.stdout.write(' #Error Occured: ' + error);
                        process.stdout.write("\n=====================================\n");
                    }
                }
                else if(rowCount < minRows){
                    total_tables_withLessThanTwoRows++;
                    process.stdout.write('\nA Table found but it does not have enough rows');
                    process.stdout.write("\n=====================================\n");
                }
                else{
                    total_hidden_tables;
                    process.stdout.write('\nTable found but it is hidden');
                    process.stdout.write("\n=====================================\n");
                }
                
            }
            i++; // This i is for the hadnles in the page 
            
        }
        
        
            
        process.stdout.write("\n\nAll tables in " + pageName + ' have been processed');
        process.stdout.write("\n#########################################################\n");
        
        // If the page has been processed and it has a table, increment the index counter (pageCounter)
        if (pageHasTableFlag == 1)
            index ++;

        total_Pages ++;
        await page.close();
    
    }
    catch(error){
        process.stdout.write(' #Error Occured: while opening the page: ' + error + '\n\n');
        total_Pages ++;
        continue;
    }
    
    }; // End of for loop 

    // Save the worksheet to the workbook
    XLSX.utils.book_append_sheet(workbook, worksheet);

    // Saving the workbook to the features.xlsx file
    XLSX.writeFile(workbook, 'features.xlsx');
    
    process.stdout.write('\nStatistics:\n');
    process.stdout.write('\nTotal Pages: ' + total_Pages);
    process.stdout.write('\nTotal Pages that has tables saved: ' + index);
    process.stdout.write('\nTotal Tables Count: ' + total_tables_count);
    process.stdout.write('\nTotal Tables Saved: ' + total_tables_saved);
    process.stdout.write('\nTotal Tables with Error: ' + total_tables_with_error);
    process.stdout.write('\nTotal Tables hidden: ' + total_hidden_tables);
    process.stdout.write('\nTotal Tables with less than two rows: ' + total_tables_withLessThanTwoRows);
    process.stdout.write('\n\nProgram Ends\n*******************************');

    await browser.close();
})(); // End of async() function. 


// ----------------------------------------------------------------------//
// This function is used to read the links in the links.txt file. 
function readLinksFromFile(filePath) {
    try {
        const data = fs.readFileSync(filePath, 'utf-8');
        const linksArray = data.split('\n').map(link => link.trim()).filter(Boolean);
        return linksArray;
    } catch (error) {
        console.error('Error reading links from file:', error);
        return [];
    }
}

// ----------------------------------------------------------------------//
// Function to get the features of the first column. 
async function getFirstCol(tableHandle) {
    const firstColumnData = await tableHandle.evaluate((table) => {
        const rows = table.rows;
        var columnWidth = rows[0].cells[0].offsetWidth;  


        var columnHeight = 0;
        for (var i = 0,row; row = rows[i]; i++){
            const first_column = row.cells[0];
                const rectObject = first_column.getBoundingClientRect();
                if(i === 0)
                {
                    columnHeight = rectObject.height;
                }
                else{
                    columnHeight +=  rectObject.height;
                }
               
        }
        return{
            height : columnHeight,
            width : columnWidth,
        };

    }, tableHandle);
    return firstColumnData;
}

// -------------------------------------------------------------------//
// Function to get the features of the first column. 
async function getLastCol(tableHandle) {
    const lastColumnData = await tableHandle.evaluate((table) => {
        const rows = table.rows;
        var last_col_Index = rows[0].cells.length - 1;
        var columnWidth = rows[0].cells[last_col_Index].offsetWidth;  


        var columnHeight = 0;
        for (var i = 0,row; row = rows[i]; i++){

            const last_column = row.cells[last_col_Index];
            const rectObject = last_column.getBoundingClientRect();
            if(i === 0)
            {
                columnHeight = rectObject.height;
            }
            else{
                columnHeight +=  rectObject.height;
            }  
        }
        return{
            height : columnHeight,
            width : columnWidth,
        };

    }, tableHandle);
    return lastColumnData;
}

// -------------------------------------------------------------------//
// A function to find if the table is leaf or no (nested or no)

async function isLeaf (tableHandle){
    const isNestedTable = await tableHandle.evaluate((table) => {
        let parentElement = table.parentElement;
  
        while (parentElement) {
          if (parentElement.tagName.toLowerCase() === 'table') {
            return 1; // The table is nested
          }
          parentElement = parentElement.parentElement;
        }
  
        return 0; // The table is not nested
      });
      return isNestedTable;
}
// -------------------------------------------------------------------//
// Function to get the features of the first row. 
async function getFirstRow(tableHandle) {
    const tableData = await tableHandle.evaluate((table) => {
        const rows = table.rows;

        const rectObject = rows[0].getBoundingClientRect();
        
        const first_cell = rows[0].cells[0];
        var computedStyle;
        if (first_cell.tagName.toLowerCase() === 'th'){
            computedStyle = getComputedStyle(first_cell);
        }
        else{
            computedStyle = getComputedStyle(rows[0]);
        }                
        
        const features = {
        height : rectObject.height,
        width : rectObject.width,
        columns : rows[0].cells.length,
        backgroundColor : computedStyle.backgroundColor,
        };

        return features;
    }, tableHandle);
    return tableData;
}
//--------------------------------------------------------------------//

// Function to get the features of the last row // 
async function getLastRow(tableHandle) {
    const tableData = await tableHandle.evaluate((table) => {
        const rows = table.rows;
        const lastRowIndex = rows.length - 1;
        const rectObject = rows[lastRowIndex].getBoundingClientRect();
        
        var computedStyle;
        computedStyle = getComputedStyle(rows[lastRowIndex]);               
        
        const features = {
        height : rectObject.height,
        width : rectObject.width,
        columns : rows[lastRowIndex].cells.length,
        backgroundColor : computedStyle.backgroundColor,
        };

        return features;
    }, tableHandle);
    return tableData;
}

//----------------------------------------------------------------//
// A function to find how many row and col spans are there 

async function findSpans(tableHandle){
    const rowColSpans = await tableHandle.evaluate((table) => {
        let rowspanCount = 0;
        let colspanCount = 0;
  
        // Iterate through all rows and cells in the table
        for (const row of table.rows) {
          for (const cell of row.cells) {
            // Check for rowspan and colspan attributes
            if (cell.rowSpan > 1) {
              rowspanCount++;
            }
            if (cell.colSpan > 1) {
              colspanCount++;
            }
          }
        }
  
        return {
          rowspanCount,
          colspanCount,
        };
      });

      return rowColSpans;
}
//----------------------------------------------------------------//
// Function to get all the text in the table, the output is then used to count the words and numbers // 
async function getData(tableHandle) {
    const tableData = await tableHandle.evaluate((table) => {
        const rows = table.rows;
        const columnsData = [];  

        for (var i = 0,row; row = rows[i]; i++){
            const columns = row.cells;
            for (var j = 0, column; column = columns[j]; j++)
            {
                columnsData.push(column.textContent.trim());
            }
        }
        
        return columnsData;
      }, tableHandle);
      return tableData;
}


//--------------------------------------------------------------//
// A function that finds how many images inside a table 
function numberOfImages(innerHTML) {
    var count = innerHTML.match(/<img/g) ? (innerHTML.match(/<img/g)).length : 0;

    return (count);
}
//--------------------------------------------------------------//
// A function that checks if there is a form element
function hasFormElement(innerHTML) {
    const formElements = '/<form|<input|<label|<select|<textarea|<button|<fieldset|<legend|<datalist|<output|<option|<optgroup/g'
    var count = (innerHTML.match(formElements) || []).length //? (innerHTML.match(/<form/g)).length : 0;

    //input|label|select|textarea|button|fieldset|legend|datalist|output|option|optgroup
    if (count)
        return 1;
    else
        return 0;
}
//--------------------------------------------------------------//
// A function that finds the number of rows inside a table 
function numberOfRows(innerHTML) {
    
    var count = (innerHTML.match(/<\/tr>/g) || []).length;

    return (count);
}
//--------------------------------------------------------------//

// A function that finds the number of columns inside a table 
// I cannot just look for how many td tags, because that's the number of cells not columns 
async function numberOfColumns(tableHandle) {
    const columnsCount = await tableHandle.evaluate((table) => {
        const rows = table.rows;
        const rows_count = table.rows.length;
        var cells_count = 0;  

        for (var i = 0,row; row = rows[i]; i++){
            const current_row_cells = row.cells.length;
            cells_count += current_row_cells;
        }
        
        return Math.floor(cells_count / rows_count);//cells_count/rows_count;
      }, tableHandle);
      return columnsCount;
}
//--------------------------------------------------------------//
// A function that finds the number of th tags inside a table 
function numberOfthTags(innerHTML) {
    var count = (innerHTML.match(/<\/th>/g) || []).length;

    return (count);
}
//--------------------------------------------------------------//

// A function that finds how many words are there within the content of a table 
function wordsFrequency(text) {
    // Join the array elements into a single string
    const joinedText = text.join();

    // Replace spaces with commas
    const textWithCommas = joinedText.replace(/\s+/g, ', ');
    
    var words = textWithCommas.match(/[a-zA-Z]+/g);

    var frequency = 0;
    if (words)
       frequency = words.length;

    return (frequency);
}
//--------------------------------------------------------------//

// A function that finds how many words are there within the content of a table 
function numbersFrequency(text) {
    // Join the array elements into a single string
    const joinedText = text.join();

    // Replace spaces with commas
    const textWithCommas = joinedText.replace(/\s+/g, ', ');
    
    // The b is to make sure it's an exact number 
    var numbers = textWithCommas.match(/\b\d+\b/g);

    var frequency = 0;
    if (numbers)
       frequency = numbers.length;

    return (frequency);
}
//--------------------------------------------------------------//