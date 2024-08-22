// Step 1    
// Set rate limiting constraints:

const requestCount = { [ip: string]: number } = {}; // Create a data structure to track request counts per IP address
const requestLimit = 10;                           // Variable to define the maximum number of requests allowed
const timeFrame = 60000;                           // Time frame variable in milliseconds (1 minute)

// Step 2
// Set rate limiter

function requestPermission(ip: string): boolean {
    // This line checks if the IP address already exists in the requestCount object.
    // Start with the negative case
    if (!requestCount[ip]) {
        // If not found, initialize the count for this IP
        requestCount[ip] = 0;
    }

    // Check if the IP address in the requestCount object has met or exceeded the rate limit
    if (requestCount[ip] >= requestLimit) {
        // If it has met the limit, send a message and return false
        console.log("Rate Limit Met");
        return false;
    }

    // In the case that the IP is in the object and has not reached the limit, increment the request count by 1 and approve the new request
    requestCount[ip] += 1;

    // Employ logic to only allow requests within the specified time window
    // After the time frame passes, decrement the count to allow future requests
    setTimeout(() => requestCount[ip] -= 1, timeFrame);

    // Return true to indicate that the request is allowed
    return true;
}

// Step 3
// Fetch data
// Create an async function 
// Asynchronously fetches user data from the API using Axios, 
// enforcing rate limits and returning a Promise that resolves to 
// an array of users.

import axios from 'axios';

async function fetchAllUsers(ip: string, baseUrl: string): Promise<any[]> {
    let allUsers: any[] = [];                       // Initialize an empty array to store user data
    let nextPageUrl: string | null = `${baseUrl}/users`; // Set the initial URL to fetch user data

    // Check if the request is allowed based on the rate limiter logic
    if (!requestPermission(ip)) {
        // If the rate limit is exceeded, throw an error to stop further execution
        throw new Error('Rate limit exceeded');
    }

    try {
        // Make a GET request to the specified URL using Axios and await the response
        const response = await axios.get(nextPageUrl);
        // Store the user data from the response in the allUsers array
        allUsers = response.data;
    } catch (error) {
        // Catch and log any errors that occur during the Axios request
        console.error('Error fetching data:', error);
    }

    // Return the array of users fetched from the API
    return allUsers;
}

// Step 4
// Process the fetched user data
// Filter the array of users to find those with a specific email domain and log the count

function processUsers(users: any[]): void {
    const targetDomain = 'example.com';              // Define the target email domain to filter by
    const count = users.filter(user => user.email.endsWith(`@${targetDomain}`)).length; // Count the users with emails matching the target domain

    // Log the number of users with the specified email domain
    console.log(`Number of users with email from ${targetDomain}:`, count);
}

// Step 5
// Main function to coordinate the entire process
// Calls fetchAllUsers and processUsers in sequence, handling any errors that arise

async function main() {
    const ipAddress = '192.168.1.1';                // Example IP address (this could be dynamic in a real-world application)
    const apiUrl = 'https://jsonplaceholder.typicode.com'; // Base URL for the API to fetch user data

    try {
        // Fetch all users by calling the fetchAllUsers function
        const users = await fetchAllUsers(ipAddress, apiUrl);
        // Process the fetched users by calling the processUsers function
        processUsers(users);
    } catch (error) {
        // Catch and log any errors that occur during the execution of the main function
        console.error('Error:', error.message);
    }
}

// Execute the main function to start the process
main();

/*
  Outline for Setting Up Rate-Limited Data Fetching and Processing with Axios

  1. Define Rate Limiting Constraints:
     - Set up a data structure to track request counts per IP address.
     - Define the maximum number of requests allowed within a specified time frame.
     - Choose an appropriate time window based on the use case.

  2. Implement Rate Limiter Logic:
     - Create a function that checks if an IP address has exceeded the allowed request limit.
     - If the limit is met, block further requests; otherwise, allow the request and increment the count.
     - Use a timing mechanism (e.g., setTimeout) to reset the count after the time window expires.

  3. Fetch Data from an API Using Axios:
     - Write an asynchronous function to retrieve data from a specified API endpoint using Axios.
     - Ensure the rate limiter is called before making the API request to enforce the constraints.
     - Handle potential errors that may occur during the data fetching process (e.g., network errors, bad responses).

  4. Process the Fetched Data:
     - Define how the fetched data should be processed, filtered, or transformed based on specific criteria.
     - Implement this logic in a separate function to keep the code modular and reusable.

  5. Coordinate the Process with a Main Function:
     - Create a main function that orchestrates the overall flow, calling the data fetching and processing functions.
     - Handle any errors that occur during the execution of the main function to ensure robustness.
     - Execute the main function to initiate the entire process.

  6. Adapt and Reuse:
     - Adjust the rate limiting, data fetching, and processing logic as needed for different applications.
     - Use this outline as a template for similar problems involving rate-limited API requests and data processing using Axios.
*/
