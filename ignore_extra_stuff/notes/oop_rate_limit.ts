// Step 1
// Define a RateLimiter class to encapsulate rate limiting logic

class RateLimiter {
    private requestCount: { [ip: string]: number } = {}; // Data structure to track request counts per IP address
    private requestLimit: number;                      // Variable to define the maximum number of requests allowed
    private timeFrame: number;                         // Time frame variable in milliseconds

    constructor(requestLimit: number, timeFrame: number) {
        this.requestLimit = requestLimit;              // Initialize the request limit
        this.timeFrame = timeFrame;                    // Initialize the time frame
    }

    // Step 2
    // Method to check if a request is permitted for a given IP address

    requestPermission(ip: string): boolean {
        // Check if the IP address already exists in the requestCount object.
        // Start with the negative case
        if (!this.requestCount[ip]) {
            // If not found, initialize the count for this IP
            this.requestCount[ip] = 0;
        }

        // Check if the IP address in the requestCount object has met or exceeded the rate limit
        if (this.requestCount[ip] >= this.requestLimit) {
            // If it has met the limit, send a message and return false
            console.log("Rate Limit Met");
            return false;
        }

        // In the case that the IP is in the object and has not reached the limit, increment the request count by 1 and approve the new request
        this.requestCount[ip] += 1;

        // Employ logic to only allow requests within the specified time window
        // After the time frame passes, decrement the count to allow future requests
        setTimeout(() => this.requestCount[ip] -= 1, this.timeFrame);

        // Return true to indicate that the request is allowed
        return true;
    }
}

// Step 3
// Define a UserFetcher class to handle API data fetching

import axios from 'axios';

class UserFetcher {
    private baseUrl: string;                          // Base URL for the API

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;                       // Initialize the base URL
    }

    // Method to fetch user data from the API using Axios

    async fetchAllUsers(ip: string, rateLimiter: RateLimiter): Promise<any[]> {
        let allUsers: any[] = [];                     // Initialize an empty array to store user data
        let nextPageUrl: string | null = `${this.baseUrl}/users`; // Set the initial URL to fetch user data

        // Check if the request is allowed based on the rate limiter logic
        if (!rateLimiter.requestPermission(ip)) {
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
}

// Step 4
// Define a function to process the fetched user data

function processUsers(users: any[]): void {
    const targetDomain = 'example.com';              // Define the target email domain to filter by
    const count = users.filter(user => user.email.endsWith(`@${targetDomain}`)).length; // Count the users with emails matching the target domain

    // Log the number of users with the specified email domain
    console.log(`Number of users with email from ${targetDomain}:`, count);
}

// Step 5
// Main function to coordinate the entire process
// Instantiate the RateLimiter and UserFetcher classes and execute the process

async function main() {
    const ipAddress = '192.168.1.1';                // Example IP address (this could be dynamic in a real-world application)
    const apiUrl = 'https://jsonplaceholder.typicode.com'; // Base URL for the API to fetch user data

    const rateLimiter = new RateLimiter(10, 60000); // Instantiate RateLimiter with requestLimit and timeFrame
    const userFetcher = new UserFetcher(apiUrl);    // Instantiate UserFetcher with the API base URL

    try {
        // Fetch all users by calling the fetchAllUsers method on the UserFetcher instance
        const users = await userFetcher.fetchAllUsers(ipAddress, rateLimiter);
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
  Outline for Setting Up Rate-Limited Data Fetching and Processing with OOP

  1. Define a RateLimiter Class:
     - Create a class to encapsulate the rate limiting logic.
     - Include properties for tracking request counts, setting request limits, and defining the time frame.
     - Implement a method that checks if an IP address has exceeded the request limit and handles incrementing/decrementing the count.

  2. Define a UserFetcher Class:
     - Create a class responsible for fetching data from an API.
     - Include a method to make an API request using Axios, enforcing rate limits by utilizing the RateLimiter class.
     - Handle potential errors that may occur during the data fetching process (e.g., network errors, bad responses).

  3. Define a Function to Process Data:
     - Create a separate function that processes the fetched data, such as filtering or transforming it based on specific criteria.
     - Keep the processing logic modular to allow for easy reuse and testing.

  4. Coordinate the Process with a Main Function:
     - Instantiate the RateLimiter and UserFetcher classes.
     - Use a main function to orchestrate the overall flow, calling the data fetching and processing methods.
     - Handle any errors that occur during the execution of the main function to ensure robustness.
     - Execute the main function to initiate the entire process.

  5. Adapt and Reuse:
     - Adjust the rate limiting, data fetching, and processing logic as needed for different applications.
     - Use this outline as a template for similar problems involving rate-limited API requests and data processing in an OOP style.
*/
